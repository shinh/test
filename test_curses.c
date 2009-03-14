#include <ncurses.h>

int main() {
    WINDOW* win = initscr();
    //start_color();
    //waddch(win, '\n');
    //waddch(win, '\n');
    //wrefresh(win);
    endwin();
    printf("%d %d %d\n", has_colors(), can_change_color(), COLOR_PAIRS);
}
