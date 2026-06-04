#pragma once
#include <glad/glad.h>

struct TextOverlay {
    GLuint program;
    GLuint vao;
    GLuint vbo;
    GLuint font_texture;
    bool initialized;

    TextOverlay();
    ~TextOverlay();

    void init();
    void draw_text(const char* text, float x, float y, float scale,
                   int screen_w, int screen_h,
                   float r, float g, float b, float a);
    void draw_panel(float x, float y, float w, float h,
                    int screen_w, int screen_h,
                    float r, float g, float b, float a);
};
