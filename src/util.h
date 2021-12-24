#ifndef UTIL_H_
#define UTIL_H_

//Enum para definir los timers
enum TIMER {
    TVAL_START,
    TVAL_RENDER_GRID,
    TVAL_SANDSIM,
    TVAL_THREAD_2,
    TVAL_THREAD_1,
    TVAL_TOTAL_2,
    TVAL_TOTAL_1
};

//util function to do simple things
int max(int a, int b);

int min(int a, int b);

void renderText(SDL_Renderer *renderer, TTF_Font *font, int r, int g, int b, char stringText[], int x, int y);

//Inicializar las fuentes para util.c
void initUtilFonts();

//Para imprimir un texto previamente formateado, solo hay que escoger la posicion
void renderFormattedText(SDL_Renderer *renderer, char stringText[], int x, int y);

//Inicia todos los timers
void startUtilTimers();

long int getTimerS(int timer);

long int getTimerMS(int timer);

void resetTimer(int timer);

#endif // RENDERING_H_