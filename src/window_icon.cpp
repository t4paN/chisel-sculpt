// Window icon: decode the embedded PNG once at startup and hand GLFW the RGBA
// pixels. Bytes are baked into the binary (include/icon_data.h) so there is no
// runtime file dependency — the AppImage/exe stays self-contained. stb_image is
// isolated to this translation unit to keep its ~280KB implementation out of the
// hot files.
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <GLFW/glfw3.h>
#include "icon_data.h"

void set_window_icon(GLFWwindow* window) {
    int w, h, channels;
    // Force 4 channels; GLFWimage wants tightly-packed RGBA.
    unsigned char* pixels = stbi_load_from_memory(
        chisel_icon_png, (int)chisel_icon_png_len, &w, &h, &channels, 4);
    if (!pixels) return; // no icon is non-fatal

    GLFWimage image;
    image.width = w;
    image.height = h;
    image.pixels = pixels;
    glfwSetWindowIcon(window, 1, &image);

    stbi_image_free(pixels);
}
