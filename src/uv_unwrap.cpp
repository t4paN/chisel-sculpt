#include "uv_unwrap.h"
#include "xatlas/xatlas.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>

uint64_t uv_topology_key(const RetopoMesh& rm) {
    // FNV-1a over the corner lists and the face boundaries: two meshes with the same
    // vertex ids in the same faces get the same key, whatever their positions.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint32_t x) { for (int i = 0; i < 4; i++) { h ^= (x >> (8 * i)) & 0xff; h *= 1099511628211ull; } };
    for (uint32_t i : rm.faces) mix(i);
    for (uint32_t i : rm.face_off) mix(i);
    return h;
}

// Ear-clip face f into triangles, as corner positions within the face (0..k-1), in the
// face's own plane (Newell normal). Quads are most of a retopo mesh, but the eraser
// merges faces into bigger, sometimes concave ones, and a fan would fold those.
static void triangulate(const RetopoMesh& rm, uint32_t f, std::vector<uint32_t>& out) {
    const uint32_t* c = rm.face(f);
    const uint32_t k = rm.face_size(f);
    Vec3 n;
    for (uint32_t j = 0; j < k; j++) {
        const Vec3& a = rm.v[c[j]];
        const Vec3& b = rm.v[c[(j + 1) % k]];
        n += Vec3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    std::vector<uint32_t> ring(k);
    for (uint32_t j = 0; j < k; j++) ring[j] = j;
    auto convex = [&](uint32_t a, uint32_t b, uint32_t d) {
        return (rm.v[c[b]] - rm.v[c[a]]).cross(rm.v[c[d]] - rm.v[c[b]]).dot(n) > 0.0f;
    };
    auto inside = [&](uint32_t p, uint32_t a, uint32_t b, uint32_t d) {
        const Vec3& P = rm.v[c[p]];
        return (rm.v[c[b]] - rm.v[c[a]]).cross(P - rm.v[c[a]]).dot(n) >= 0.0f &&
               (rm.v[c[d]] - rm.v[c[b]]).cross(P - rm.v[c[b]]).dot(n) >= 0.0f &&
               (rm.v[c[a]] - rm.v[c[d]]).cross(P - rm.v[c[d]]).dot(n) >= 0.0f;
    };
    while (ring.size() > 3) {
        const size_t m = ring.size();
        size_t ear = m;
        for (size_t i = 0; i < m && ear == m; i++) {
            const uint32_t a = ring[(i + m - 1) % m], b = ring[i], d = ring[(i + 1) % m];
            if (!convex(a, b, d)) continue;
            bool clear = true;
            for (size_t j = 0; j < m && clear; j++) {
                const uint32_t p = ring[j];
                if (p != a && p != b && p != d && inside(p, a, b, d)) clear = false;
            }
            if (clear) ear = i;
        }
        if (ear == m) ear = 0;   // nothing clean (a twisted face): clip anyway, it still covers it
        out.push_back(ring[(ear + m - 1) % m]);
        out.push_back(ring[ear]);
        out.push_back(ring[(ear + 1) % m]);
        ring.erase(ring.begin() + ear);
    }
    out.insert(out.end(), ring.begin(), ring.end());
}

const char* uv_unwrap(const RetopoMesh& rm, const UvSettings& s, UvLayout& out, double& ms) {
    out.clear();
    ms = 0.0;
    const uint32_t nf = rm.face_count();
    if (nf == 0) return "no retopo mesh to unwrap (Build one in mode 0 first)";
    const auto t0 = std::chrono::steady_clock::now();

    // xatlas takes polygons too, but that path mis-indexes faces it rejects as degenerate
    // (it writes them as if every face were a triangle) and can still split one quad's
    // triangles between islands. So it gets triangles, and the quads are put back below.
    // tri_corner holds, per triangle corner, which corner of rm.faces it came from.
    std::vector<uint32_t> tri_idx, tri_corner, tri_face, local;
    for (uint32_t f = 0; f < nf; f++) {
        if (rm.face_size(f) < 3) continue;
        local.clear();
        triangulate(rm, f, local);
        for (uint32_t j : local) {
            tri_corner.push_back(rm.face_off[f] + j);
            tri_idx.push_back(rm.faces[rm.face_off[f] + j]);
        }
        for (size_t t = 0; t < local.size() / 3; t++) tri_face.push_back(f);
    }
    const uint32_t nt = (uint32_t)tri_face.size();

    // ---- Pass 1: xatlas cuts the mesh into islands and flattens each one.
    xatlas::MeshDecl decl;
    decl.vertexPositionData   = rm.v.data();
    decl.vertexPositionStride = sizeof(Vec3);
    decl.vertexCount          = (uint32_t)rm.v.size();
    decl.indexData            = tri_idx.data();
    decl.indexFormat          = xatlas::IndexFormat::UInt32;
    decl.indexCount           = (uint32_t)tri_idx.size();
    // No normals: rm.n is the sculpt's normal under each vertex, which can disagree with
    // the low-poly face it's on. Seams should follow the low-poly surface.
    xatlas::Atlas* atlas = xatlas::Create();
    const xatlas::AddMeshError err = xatlas::AddMesh(atlas, decl);
    if (err != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        std::printf("[uv] xatlas refused the mesh: %s\n", xatlas::StringForEnum(err));
        return "the unwrapper refused the mesh (see the console)";
    }
    xatlas::ChartOptions co;
    co.maxCost = s.max_cost;
    co.fixWinding = true;
    // Only the islands' shapes are kept from this pass, so its packing is the default
    // quick one; pass 2 packs for real once the islands are repaired.
    xatlas::Generate(atlas, co, xatlas::PackOptions());
    if (atlas->meshCount != 1 || atlas->meshes[0].indexCount != tri_idx.size()) {
        xatlas::Destroy(atlas);
        return "the unwrapper produced nothing";
    }
    xatlas::Mesh& m = atlas->meshes[0];

    // fixWinding looks at one face per island and still hands back whole islands mirrored
    // (half of them on a real retopo, 2026-10-26; never part of one). So count each
    // island's backwards triangles and mirror the island back across its own box.
    std::vector<uint32_t> neg(m.chartCount, 0), tot(m.chartCount, 0);
    std::vector<float> u_lo(m.chartCount, 1e30f), u_hi(m.chartCount, -1e30f);
    for (uint32_t t = 0; t < nt; t++) {
        const xatlas::Vertex& a = m.vertexArray[m.indexArray[3 * t]];
        const xatlas::Vertex& b = m.vertexArray[m.indexArray[3 * t + 1]];
        const xatlas::Vertex& d = m.vertexArray[m.indexArray[3 * t + 2]];
        if (a.chartIndex < 0) continue;
        const float area = (b.uv[0] - a.uv[0]) * (d.uv[1] - a.uv[1]) - (d.uv[0] - a.uv[0]) * (b.uv[1] - a.uv[1]);
        tot[a.chartIndex]++;
        neg[a.chartIndex] += area < 0.0f;
    }
    for (uint32_t v = 0; v < m.vertexCount; v++) {
        const int32_t c = m.vertexArray[v].chartIndex;
        if (c < 0) continue;
        u_lo[c] = std::min(u_lo[c], m.vertexArray[v].uv[0]);
        u_hi[c] = std::max(u_hi[c], m.vertexArray[v].uv[0]);
    }
    uint32_t mirrored = 0;
    for (uint32_t c = 0; c < m.chartCount; c++) mirrored += 2 * neg[c] > tot[c];
    for (uint32_t v = 0; v < m.vertexCount; v++) {
        const int32_t c = m.vertexArray[v].chartIndex;
        if (c >= 0 && 2 * neg[c] > tot[c]) m.vertexArray[v].uv[0] = u_lo[c] + u_hi[c] - m.vertexArray[v].uv[0];
    }

    // Each face goes to the island most of its triangles landed in; its corners take their
    // UVs from those triangles. A rejected (degenerate) triangle has no island. `flat` is
    // in pass 1's units: one scale for every island, which is all pass 2 needs.
    std::vector<float> flat(rm.faces.size() * 2, 0.0f);
    std::vector<uint8_t> have(rm.faces.size(), 0);
    out.face_chart.assign(nf, ~0u);
    uint32_t split = 0, unfolded = 0, orphans = 0;
    for (uint32_t t = 0, f = 0; f < nf; f++) {
        const uint32_t t0f = t;
        while (t < nt && tri_face[t] == f) t++;
        uint32_t best = ~0u, best_n = 0;
        for (uint32_t a = t0f; a < t; a++) {
            const int32_t ca = m.vertexArray[m.indexArray[3 * a]].chartIndex;
            if (ca < 0) continue;
            uint32_t n = 0;
            for (uint32_t b = t0f; b < t; b++) n += m.vertexArray[m.indexArray[3 * b]].chartIndex == ca;
            if (n > best_n) { best_n = n; best = (uint32_t)ca; }
        }
        if (best == ~0u) { orphans++; continue; }
        if (best_n != t - t0f) split++;
        out.face_chart[f] = best;
        for (uint32_t a = 3 * t0f; a < 3 * t; a++) {
            const xatlas::Vertex& v = m.vertexArray[m.indexArray[a]];
            if (v.chartIndex != (int32_t)best) continue;
            // No v flip: v up (OBJ) is how xatlas keeps a face's winding; flipping it
            // would mirror every island.
            flat[2 * tri_corner[a]]     = v.uv[0];
            flat[2 * tri_corner[a] + 1] = v.uv[1];
            have[tri_corner[a]] = 1;
        }
        // A corner only the off-island triangles reached: unfold it into this island across
        // its two neighbours, keeping the 3D triangle's shape and the island's scale.
        // Repeat for an n-gon that lost more than one. Pass 2 makes room for it.
        const uint32_t o = rm.face_off[f], k = rm.face_size(f);
        for (bool again = true; again;) {
            again = false;
            for (uint32_t j = 0; j < k; j++) {
                const uint32_t c = o + j, cp = o + (j + k - 1) % k, cn = o + (j + 1) % k;
                if (have[c] || !have[cp] || !have[cn]) continue;
                const Vec3 P = rm.v[rm.faces[cp]], N = rm.v[rm.faces[cn]], X = rm.v[rm.faces[c]];
                const Vec3 e = N - P;
                const float L3 = e.length();
                const float eu = flat[2 * cn] - flat[2 * cp], ev = flat[2 * cn + 1] - flat[2 * cp + 1];
                if (L3 < 1e-12f || eu * eu + ev * ev < 1e-24f) continue;
                // X in the edge's frame: along it, and across it.
                const float along = (X - P).dot(e) / (L3 * L3);
                const float across = (X - P - e * along).length() / L3;
                // Corners run counter-clockwise in UV, so going cp -> c -> cn, X sits to
                // the right of the cp -> cn edge.
                flat[2 * c]     = flat[2 * cp]     + eu * along + ev * across;
                flat[2 * c + 1] = flat[2 * cp + 1] + ev * along - eu * across;
                have[c] = 1; unfolded++; again = true;
            }
        }
    }
    const uint32_t charts1 = m.chartCount;
    xatlas::Destroy(atlas);
    // Islands numbered 0..n-1 again: one whose every face went to a neighbour is gone.
    {
        std::vector<uint32_t> renum(charts1, ~0u);
        for (uint32_t& c : out.face_chart)
            if (c != ~0u) { if (renum[c] == ~0u) renum[c] = out.charts++; c = renum[c]; }
    }

    // ---- Pass 2: pack the repaired islands into the square, with padding. xatlas finds
    // the islands by shared UV corners; one vertex per (island, mesh vertex, UV) keeps a
    // seam vertex's two copies apart, and the island as material keeps islands from
    // merging where they happen to touch.
    std::vector<float> pv;
    std::vector<uint32_t> pidx(tri_idx.size()), pmat(nt);
    {
        std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> seen;
        for (uint32_t i = 0; i < tri_idx.size(); i++) {
            const uint32_t c = tri_corner[i], isl = out.face_chart[tri_face[i / 3]];
            uint32_t ub, vb;
            std::memcpy(&ub, &flat[2 * c], 4);
            std::memcpy(&vb, &flat[2 * c + 1], 4);
            const auto key = std::make_tuple(isl, tri_idx[i], ub, vb);
            auto it = seen.find(key);
            if (it == seen.end()) {
                it = seen.emplace(key, (uint32_t)(pv.size() / 2)).first;
                pv.push_back(flat[2 * c]);
                pv.push_back(flat[2 * c + 1]);
            }
            pidx[i] = it->second;
        }
        for (uint32_t t = 0; t < nt; t++) pmat[t] = out.face_chart[tri_face[t]];
    }
    xatlas::UvMeshDecl ud;
    ud.vertexUvData     = pv.data();
    ud.vertexStride     = 2 * sizeof(float);
    ud.vertexCount      = (uint32_t)(pv.size() / 2);
    ud.indexData        = pidx.data();
    ud.indexFormat      = xatlas::IndexFormat::UInt32;
    ud.indexCount       = (uint32_t)pidx.size();
    ud.faceMaterialData = pmat.data();
    atlas = xatlas::Create();
    if (xatlas::AddUvMesh(atlas, ud) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        out.clear();
        return "the packer refused the islands";
    }
    xatlas::PackOptions po;
    po.resolution = (uint32_t)s.resolution;
    po.padding    = (uint32_t)s.padding;
    xatlas::Generate(atlas, xatlas::ChartOptions(), po);   // on a UV mesh: find its islands, then pack

    const char* why = nullptr;
    float used = 0.0f;
    uint32_t turned = 0, dotted = 0;
    if (atlas->meshCount != 1 || atlas->width == 0 || atlas->height == 0 || atlas->meshes[0].indexCount != pidx.size()) {
        why = "the packer produced nothing";
    } else if (atlas->atlasCount > 1) {
        // Fixed resolution with an estimated scale should always fit one square; more
        // than one means the islands didn't, and a second sheet has nowhere to go.
        why = "the islands didn't fit one texture";
    } else {
        const xatlas::Mesh& pm = atlas->meshes[0];
        used = atlas->utilization ? atlas->utilization[0] : 0.0f;
        const float iw = 1.0f / (float)atlas->width, ih = 1.0f / (float)atlas->height;
        out.uv.assign(rm.faces.size() * 2, 0.0f);
        std::fill(have.begin(), have.end(), 0);
        for (uint32_t i = 0; i < pidx.size(); i++) {
            const xatlas::Vertex& v = pm.vertexArray[pm.indexArray[i]];
            if (v.chartIndex < 0) continue;   // a triangle too thin to pack: its corners come from the others
            out.uv[2 * tri_corner[i]]     = v.uv[0] * iw;
            out.uv[2 * tri_corner[i] + 1] = v.uv[1] * ih;
            have[tri_corner[i]] = 1;
        }
        // The packer turns some islands a quarter turn by swapping u and v, which mirrors
        // them (likely what pass 1's mirrored islands were too). Same cure: per island,
        // signed area over its faces; mostly backwards = mirror it across its own box.
        {
            std::vector<double> area(out.charts, 0.0);
            std::vector<float> lo(out.charts, 1e30f), hi(out.charts, -1e30f);
            for (uint32_t f = 0; f < nf; f++) {
                const uint32_t c = out.face_chart[f];
                if (c == ~0u) continue;
                const uint32_t o = rm.face_off[f], k = rm.face_size(f);
                for (uint32_t j = 0; j < k; j++) {
                    const uint32_t a = o + j, b = o + (j + 1) % k;
                    area[c] += (double)out.uv[2 * a] * out.uv[2 * b + 1] - (double)out.uv[2 * b] * out.uv[2 * a + 1];
                    if (have[a]) { lo[c] = std::min(lo[c], out.uv[2 * a]); hi[c] = std::max(hi[c], out.uv[2 * a]); }
                }
            }
            for (uint32_t f = 0; f < nf; f++) {
                const uint32_t c = out.face_chart[f];
                if (c == ~0u || area[c] >= 0.0) continue;
                for (uint32_t j = rm.face_off[f]; j < rm.face_off[f + 1]; j++) out.uv[2 * j] = lo[c] + hi[c] - out.uv[2 * j];
            }
            for (uint32_t c = 0; c < out.charts; c++) turned += area[c] < 0.0;
        }
        // Whatever no packed triangle reached (a face in no island, or a corner only a
        // sliver touched) takes its face's first placed corner: a dot, not a stray line.
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t o = rm.face_off[f], k = rm.face_size(f);
            uint32_t src = ~0u;
            for (uint32_t j = 0; j < k && src == ~0u; j++) if (have[o + j]) src = o + j;
            for (uint32_t j = 0; j < k; j++) dotted += !have[o + j];
            for (uint32_t j = 0; j < k; j++)
                if (!have[o + j] && src != ~0u) { out.uv[2 * (o + j)] = out.uv[2 * src]; out.uv[2 * (o + j) + 1] = out.uv[2 * src + 1]; }
        }
        out.key = uv_topology_key(rm);
    }
    xatlas::Destroy(atlas);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (why) { out.clear(); return why; }
    std::printf("[uv] unwrap: %u faces -> %u islands, %.0f%% of the square used, %.1f ms"
                " (mirrored back %u + %u after packing, split faces %u, corners unfolded %u, faces in no island %u, corners dotted %u)\n",
                nf, out.charts, used * 100.0f, ms, mirrored, turned, split, unfolded, orphans, dotted);
    return nullptr;
}
