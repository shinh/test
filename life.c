#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define W 300
#define H 300
int get(int* field, int y, int x){
    return field[x%W+y%H*W];
}
int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Surface* scr = SDL_SetVideoMode(800, 800, 32, SDL_SWSURFACE);
    int field[W*H], field_next[W*H];
    int i,j;
    srand(time(NULL));
    for (i = 0; i < H; i++) {
        for (j = 0; j < W; j++) {
            field[j+i*W] = rand()%100<40;
        }
    }
    int done = 0;
    while (!done) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_KEYDOWN) {
                if (ev.key.keysym.sym == SDLK_ESCAPE) {
                    done = 1;
                }
            }
        }

        SDL_FillRect(scr, NULL, SDL_MapRGB(scr->format, 0,0,0));
        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                if (field[j+i*W]) {
                    SDL_Rect rect;
                    rect.x = j*2;
                    rect.y = i*2;
                    rect.w = 2;
                    rect.h = 2;
                    SDL_FillRect(scr, &rect,
                                 SDL_MapRGB(scr->format, 255,255,255));
                }
            }
        }

        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                int v=
                    get(field,i-1,j-1)+get(field,i-1,j)+get(field,i-1,j+1)+
                    get(field,i,j-1)+get(field,i,j)+get(field,i,j+1)+
                    get(field,i+1,j-1)+get(field,i+1,j)+get(field,i+1,j+1);
                field_next[j+i*W] = v==3||v==4&&get(field,i,j)==1;
            }
        }

        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                field[j+i*W]=field_next[j+i*W];
            }
        }

        SDL_Flip(scr);
        SDL_Delay(100);
    }
    SDL_Quit();
}
