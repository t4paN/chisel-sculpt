// Trimmed WinTab packet-definition header. Define PACKETDATA (an OR of PK_* bits)
// and PACKETMODE before including this; the PACKET struct is then emitted with
// exactly those fields, in ascending PK-bit order — which is the order
// WTPacketsGet writes them. Mirrors the codegen scheme of Wacom's pktdef.h but
// covers only the fields Chisel may request. Requires wintab.h first.
//
// This header is intentionally re-includable with different PACKETDATA (no guard).

#ifndef PACKETDATA
#define PACKETDATA (PK_CURSOR | PK_NORMAL_PRESSURE)
#endif
#ifndef PACKETMODE
#define PACKETMODE 0
#endif

typedef struct tagPACKET {
#if (PACKETDATA & PK_CONTEXT)
    HCTX  pkContext;
#endif
#if (PACKETDATA & PK_STATUS)
    UINT  pkStatus;
#endif
#if (PACKETDATA & PK_TIME)
    DWORD pkTime;
#endif
#if (PACKETDATA & PK_CHANGED)
    WTPKT pkChanged;
#endif
#if (PACKETDATA & PK_SERIAL_NUMBER)
    UINT  pkSerialNumber;
#endif
#if (PACKETDATA & PK_CURSOR)
    UINT  pkCursor;
#endif
#if (PACKETDATA & PK_BUTTONS)
    DWORD pkButtons;
#endif
#if (PACKETDATA & PK_X)
    LONG  pkX;
#endif
#if (PACKETDATA & PK_Y)
    LONG  pkY;
#endif
#if (PACKETDATA & PK_Z)
    LONG  pkZ;
#endif
#if (PACKETDATA & PK_NORMAL_PRESSURE)
    UINT  pkNormalPressure;
#endif
#if (PACKETDATA & PK_TANGENT_PRESSURE)
    UINT  pkTangentPressure;
#endif
} PACKET, *PPACKET, *LPPACKET;
