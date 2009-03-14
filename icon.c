#include <SDL.h>
int conv1(int v) {
    return v < 48 ? 0 : v < 115 ? 1 : v < 155 ? 2 : v < 195 ? 3 : 235 ? 4 : 5;
}
int conv(int r, int g, int b) {
    return 16 + conv1(r) * 36 + conv1(g) * 6 + conv1(b);
}
int main() {
    //SDL_Init(SDL_INIT_VIDEO);
    SDL_Surface* surf = SDL_LoadBMP("icon.bmp");
    //printf("%d\n", surf->format->BytesPerPixel);
    Uint8* p = surf->pixels;
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            Uint8 v = *(Uint8*)(p+i*surf->pitch+j);
            Uint8 r, g, b;
            SDL_GetRGB(v, surf->format, &r, &g, &b);
            //printf("%02x %02x %02x\n", r, g, b);
            printf("\x1b[48;5;%dm  ", conv(r,g,b));
        }
        puts(i == 31 ? "\x1b[0m" : "");
    }
}
