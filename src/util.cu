#include <sys/time.h>
#include <SDL2/SDL.h>
//#include "SDL_ttf.h"
#include "/usr/include/SDL2/SDL_ttf.h"
#include "util.h"

//Timers
struct timeval tval_start, tval_render_grid, tval_sandsim, tval_threads_1, tval_threads_2, tval_total_1, tval_total_2;;

//Fuentes
TTF_Font *cfont;

/*int max(int x,int k)
{
	
	if(x>k)
    return x;
  else
    return k;
}

int min(int x,int k)
{
	
	if(x<k)
    return x;
  else
    return k;
}*/

//Para imprimir un texto con opción de escoger la fuente y la posicion
void renderText(SDL_Renderer *renderer, TTF_Font *font, int r, int g, int b, char stringText[], int x, int y){
        SDL_Color font_color = { .r = r, .g = g, .b = b };
        //render text on screen with SDL with the element that is being drawn 
        SDL_Surface *surfaceMessage = TTF_RenderText_Solid(font, stringText, font_color); //
        SDL_Texture *Message = SDL_CreateTextureFromSurface(renderer, surfaceMessage); // now you can convert it into a texture
        SDL_Rect Message_rect;
        Message_rect.x = x;
        Message_rect.y = y;
        Message_rect.w = surfaceMessage->w;
        Message_rect.h = surfaceMessage->h;
        SDL_RenderCopy(renderer, Message, NULL, &Message_rect);
        //SDL_FreeSurface(surfaceMessage);
        //SDL_DestroyTexture(Message);
}


//Inicializar las fuentes para util.c
void initUtilFonts(){
  cfont=TTF_OpenFont("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", 16);
  if(!cfont) {
    printf("TTF_OpenFont: %s\n", TTF_GetError());
    // handle error
  }
}

//Para imprimir un texto previamente formateado, solo hay que escoger la posicion
void renderFormattedText(SDL_Renderer *renderer, char stringText[], int x, int y){
  // load font.ttf at size 16 into font
  if(cfont == NULL){ //Sólo inicializa las fuentes una vez
    initUtilFonts();
  }
  renderText(renderer,cfont,6, 150, 78, stringText, x, y);
}

//Inicia todos los timers
void startUtilTimers(){
  resetTimer(TVAL_START);
  resetTimer(TVAL_RENDER_GRID);
  resetTimer(TVAL_SANDSIM);
  resetTimer(TVAL_THREAD_2);
  resetTimer(TVAL_THREAD_1);
  resetTimer(TVAL_TOTAL_2);
  resetTimer(TVAL_TOTAL_1);
}

long int getTimerS(int timer){
  switch (timer)
  {
  case TVAL_START:
    return (long int) tval_start.tv_sec;
  case TVAL_RENDER_GRID:
    return (long int) tval_render_grid.tv_sec;
  case TVAL_SANDSIM:
    return (long int) tval_sandsim.tv_sec;
  case TVAL_THREAD_2:
    return (long int) tval_threads_2.tv_sec;
  case TVAL_THREAD_1:
    return (long int) tval_threads_1.tv_sec;
  case TVAL_TOTAL_2:
    return (long int) tval_total_2.tv_sec;
  case TVAL_TOTAL_1:
    return (long int) tval_total_1.tv_sec;
  default:
    break;
  }
  return -1;
}

long int getTimerMS(int timer){
  switch (timer)
  {
  case TVAL_START:
    return (long int) tval_start.tv_usec;
  case TVAL_RENDER_GRID:
    return (long int) tval_render_grid.tv_usec;
  case TVAL_SANDSIM:
    return (long int) tval_sandsim.tv_usec;
  case TVAL_THREAD_2:
    return (long int)tval_threads_2.tv_usec;
  case TVAL_THREAD_1:
    return (long int) tval_threads_1.tv_usec;
  case TVAL_TOTAL_2:
    return (long int) tval_total_2.tv_usec;
  case TVAL_TOTAL_1:
    return (long int) tval_total_1.tv_usec;
  default:
    break;
  }
  return -1;
}

void resetTimer(int timer){
  switch (timer)
  {
  case TVAL_START:
    gettimeofday(&tval_start, NULL);
    break;
  case TVAL_RENDER_GRID:
    gettimeofday(&tval_render_grid, NULL);
    break;
  case TVAL_SANDSIM:
    gettimeofday(&tval_sandsim, NULL);
    break;
  case TVAL_THREAD_2:
    gettimeofday(&tval_threads_2, NULL);
    break;
  case TVAL_THREAD_1:
    gettimeofday(&tval_threads_1, NULL);
    break;
  case TVAL_TOTAL_2:
    gettimeofday(&tval_total_2, NULL);
    break;
  case TVAL_TOTAL_1:
    gettimeofday(&tval_total_1, NULL);
    break;
  default:
    break;
  }
}