#include <stdio.h>
#include <stdlib.h>

#include <SDL.h>
#include <SDL_image.h>

Uint32 getpixel(SDL_Surface *surface, int x, int y) {
    int bpp = surface->format->BytesPerPixel;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 1:
        return *p;

    case 2:
        return *(Uint16 *)p;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
            return p[0] << 16 | p[1] << 8 | p[2];
        else
            return p[0] | p[1] << 8 | p[2] << 16;
    case 4:
        return *(Uint32 *)p;

    default:
        return 0;
    }
}

void putpixel(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    int bpp = surface->format->BytesPerPixel;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 1:
        *p = pixel;
        break;

    case 2:
        *(Uint16 *)p = pixel;
        break;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            p[0] = (pixel >> 16) & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = pixel & 0xff;
        } else {
            p[0] = pixel & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = (pixel >> 16) & 0xff;
        }
        break;

    case 4:
        *(Uint32 *)p = pixel;
        break;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <img1> <img2> <out_img>\n", argv[0]);
        exit(1);
    }

    SDL_Surface* s1 = IMG_Load(argv[1]);
    SDL_Surface* s2 = IMG_Load(argv[2]);
    if (s1->w != s2->w) {
        printf("width differ\n");
        exit(1);
    }
    if (s1->h != s2->h) {
        printf("height differ\n");
        exit(1);
    }

    SDL_Surface* o = SDL_CreateRGBSurface(SDL_SWSURFACE, s1->w, s1->h, 32,
                                          0, 0, 0, 0);
    //o = s2;

    int x, y;
    int tot_cnt = 0, err_cnt = 0;
    for (y = 0; y < s1->h; y++) {
        for (x = 0; x < s1->w; x++) {
            Uint32 p1 = getpixel(s1, x, y);
            Uint32 p2 = getpixel(s2, x, y);
            Uint8 r1, g1, b1;
            SDL_GetRGB(p1, s1->format, &r1, &g1, &b1);
            Uint8 r2, g2, b2;
            SDL_GetRGB(p2, s2->format, &r2, &g2, &b2);

            int diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2);

            if (diff)
                err_cnt++;
            tot_cnt++;

            int c = diff / 3;
            putpixel(o, x, y, SDL_MapRGB(o->format, c, c, c));
#if 0
            if (c > 127) {
                SDL_Rect r;
                r.x = x - 5; r.y = y - 5; r.w = 10; r.h = 10;
                SDL_FillRect(o, &r, SDL_MapRGB(o->format, c, 0, 0));
                //putpixel(o, x, y, SDL_MapRGB(o->format, c, 0, 0));
            } else {
                c = 255 - (r2 + g2 + b2) / 30;
                putpixel(o, x, y, SDL_MapRGB(o->format, c, c, c));
            }
#endif
        }
    }

    printf("%d/%d differ\n", err_cnt, tot_cnt);

    SDL_SaveBMP(o, argv[3]);
}
