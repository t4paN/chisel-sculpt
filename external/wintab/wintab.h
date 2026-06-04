#ifndef __WINTAB_H__
#define __WINTAB_H__

// Trimmed, self-contained WinTab client header for Chisel's pen-pressure path.
// Only the types/constants Chisel actually uses are declared. The struct layouts
// (LOGCONTEXTA, AXIS) are ABI-exact copies of Wacom's public WinTab SDK so that
// Wintab32.dll reads/writes them correctly. Requires <windows.h> to be included
// first (HWND, UINT, LONG, DWORD, BOOL, HANDLE, DECLARE_HANDLE).
//
// We never link wintab32.lib — the entry points are GetProcAddress'd from the
// runtime DLL (see src/tablet.cpp), so these are types/constants only.

#ifdef __cplusplus
extern "C" {
#endif

DECLARE_HANDLE(HCTX);

typedef DWORD WTPKT;   // packet-field bitmask
typedef DWORD FIX32;   // 16.16 fixed point

#define LCNAMELEN 40

typedef struct tagAXIS {
    LONG  axMin;
    LONG  axMax;
    UINT  axUnits;
    FIX32 axResolution;
} AXIS, *PAXIS, *LPAXIS;

typedef struct tagLOGCONTEXTA {
    char  lcName[LCNAMELEN];
    UINT  lcOptions;
    UINT  lcStatus;
    UINT  lcLocks;
    UINT  lcMsgBase;
    UINT  lcDevice;
    UINT  lcPktRate;
    WTPKT lcPktData;
    WTPKT lcPktMode;
    WTPKT lcMoveMask;
    DWORD lcBtnDnMask;
    DWORD lcBtnUpMask;
    LONG  lcInOrgX;
    LONG  lcInOrgY;
    LONG  lcInOrgZ;
    LONG  lcInExtX;
    LONG  lcInExtY;
    LONG  lcInExtZ;
    LONG  lcOutOrgX;
    LONG  lcOutOrgY;
    LONG  lcOutOrgZ;
    LONG  lcOutExtX;
    LONG  lcOutExtY;
    LONG  lcOutExtZ;
    FIX32 lcSensX;
    FIX32 lcSensY;
    FIX32 lcSensZ;
    BOOL  lcSysMode;
    int   lcSysOrgX;
    int   lcSysOrgY;
    int   lcSysExtX;
    int   lcSysExtY;
    FIX32 lcSysSensX;
    FIX32 lcSysSensY;
} LOGCONTEXTA, *PLOGCONTEXTA, *LPLOGCONTEXTA;

// --- WTInfo categories / indices ---
#define WTI_DEFSYSCTX 4
#define WTI_DEVICES   100
#define DVC_NPRESSURE 15

// --- lcOptions bits ---
#define CXO_SYSTEM   0x0001
#define CXO_PEN      0x0002
#define CXO_MESSAGES 0x0004

// --- packet-field (PK_*) bits, ascending order = packet field order ---
#define PK_CONTEXT          0x0001
#define PK_STATUS           0x0002
#define PK_TIME             0x0004
#define PK_CHANGED          0x0008
#define PK_SERIAL_NUMBER    0x0010
#define PK_CURSOR           0x0020
#define PK_BUTTONS          0x0040
#define PK_X                0x0080
#define PK_Y                0x0100
#define PK_Z                0x0200
#define PK_NORMAL_PRESSURE  0x0400
#define PK_TANGENT_PRESSURE 0x0800
#define PK_ORIENTATION      0x1000
#define PK_ROTATION         0x2000

// Entry-point signatures (for the typedefs in tablet.cpp); not imported here.
UINT  WTInfoA(UINT, UINT, LPVOID);
HCTX  WTOpenA(HWND, LPLOGCONTEXTA, BOOL);
BOOL  WTClose(HCTX);
int   WTPacketsGet(HCTX, int, LPVOID);
BOOL  WTEnable(HCTX, BOOL);

#ifdef __cplusplus
}
#endif

#endif // __WINTAB_H__
