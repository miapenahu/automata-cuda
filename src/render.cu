#include <SDL2/SDL.h>
#include "/usr/include/SDL2/SDL_ttf.h"
#include <stdbool.h> 
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
//CUDA imports
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "logic.h"
#include "render.h"
#include "util.h"


int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

//!! define color of the automate, already listed in logic properties

const SDL_Color BLACK_CELL_COLOR = { .r = 0, .g = 0, .b = 0 };
const SDL_Color WHITE_CELL_COLOR = { .r = 255, .g = 255, .b = 255 };
const SDL_Color BLUE_CELL_COLOR = { .r = 0, .g = 0, .b = 255 };
const SDL_Color RED_CELL_COLOR = { .r = 255, .g = 0, .b = 0 };
const SDL_Color GRAY_CELL_COLOR = { .r = 128, .g = 128, .b = 128 };
const SDL_Color YELLOW_CELL_COLOR = { .r = 255, .g = 255, .b = 0 };
const SDL_Color PURPLE_CELL_COLOR = { .r = 119, .g = 0, .b = 199 };
const SDL_Color GREEN_CELL_COLOR = { .r = 0, .g = 255, .b = 0 };
const SDL_Color WHITEBLUE_CELL_COLOR = { .r = 176, .g = 241, .b = 247 };
const SDL_Color GRAYSMOKE_CELL_COLOR = { .r = 176, .g = 176, .b = 176 };
const SDL_Color STRUCTURE_CELL_COLOR = { .r = 67, .g = 59, .b = 27 };
const SDL_Color ANT_COLOR = { .r = 255, .g = 50, .b = 50 };


//NUMBER OF THREADS WE ARE USING IN SPECIFIC MOMENT
int blockSize=1024;
int gridSize=0;
int s_cambio_threads = 10; //Segundos para aumentar las thread

//Variables benchmarking Render Grid
int render_grid_framecnt = 0;
int fps_render_grid = 0;
double avg_time_render_grid = 0;

void render_grid(SDL_Renderer *renderer, const state_t *state)
{
    //* Change thread number every 5 seconds and check if threads are less than the maximum of threads, so we can see the FPS for each number of threads
    ++render_grid_framecnt; //Ir sumando los frames

    resetTimer(TVAL_THREAD_2); //Actualizar timer
    
    if(( getTimerS(TVAL_THREAD_2)-getTimerS(TVAL_THREAD_1) >= s_cambio_threads) && (blockSize >= BLOCKSIZE)){
        
        double avg_fps = render_grid_framecnt/s_cambio_threads;
        long double avg_time = ((avg_time_render_grid/avg_fps)/s_cambio_threads)*1000000;
        fps_render_grid = avg_fps;

        int nElem = N * N;
        gridSize = (nElem + blockSize - 1) / blockSize;

        SDL_Log("[RENDER] : kernels: <<<%d,%d>>>, #de FPS promedio de los anteriores %d segundos: %0.1f, Tiempo promedio (us): %0.1Lf", 
        gridSize,
        blockSize, 
        s_cambio_threads, 
        avg_fps,
        avg_time);
        blockSize -= 128; //reduce el tamaño del bloque
        resetTimer(TVAL_THREAD_2); //Actualizar timer
        resetTimer(TVAL_THREAD_1); //Actualizar timer
        render_grid_framecnt = 0; //Reiniciar la cuenta
        avg_time_render_grid = 0; //Reniciar el promedio de tiempo en X segundos
       
    }
    
    //#pragma omp parallel num_threads(threads)  
    //{
    //#pragma omp for
    for (int x = 0; x < N; x++){  
        for (int y = 0; y < N; y++) {
              SDL_Rect rect = {
                  .x = x * CELL_WIDTH,
                  .y = y * CELL_HEIGHT,
                  .w = CELL_WIDTH,
                  .h = CELL_HEIGHT
              };
              int idx = (y * N) + x;
              switch(state->board[idx]) {
                  case BLACK:
                      SDL_SetRenderDrawColor(renderer, BLACK_CELL_COLOR.r, BLACK_CELL_COLOR.g, BLACK_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;

                  case BLUE:
                      SDL_SetRenderDrawColor(renderer, BLUE_CELL_COLOR.r, BLUE_CELL_COLOR.g, BLUE_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;

                  case RED:
                      SDL_SetRenderDrawColor(renderer, RED_CELL_COLOR.r, RED_CELL_COLOR.g, RED_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  case GRAY:
                      SDL_SetRenderDrawColor(renderer, GRAY_CELL_COLOR.r, GRAY_CELL_COLOR.g, GRAY_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  case YELLOW:
                      SDL_SetRenderDrawColor(renderer, YELLOW_CELL_COLOR.r, YELLOW_CELL_COLOR.g, YELLOW_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  case WHITEBLUE:
                      SDL_SetRenderDrawColor(renderer, WHITEBLUE_CELL_COLOR.r, WHITEBLUE_CELL_COLOR.g, WHITEBLUE_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  case GREEN:
                      SDL_SetRenderDrawColor(renderer, GREEN_CELL_COLOR.r, GREEN_CELL_COLOR.g, GREEN_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                    
                  case PURPLE:
                      SDL_SetRenderDrawColor(renderer, PURPLE_CELL_COLOR.r, PURPLE_CELL_COLOR.g, PURPLE_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  case GRAYSMOKE:
                      SDL_SetRenderDrawColor(renderer, GRAYSMOKE_CELL_COLOR.r, GRAYSMOKE_CELL_COLOR.g, GRAYSMOKE_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  case STRUCTURE:
                      SDL_SetRenderDrawColor(renderer, STRUCTURE_CELL_COLOR.r, STRUCTURE_CELL_COLOR.g, STRUCTURE_CELL_COLOR.b, 255);
                      SDL_RenderFillRect(renderer, &rect);
                      break;
                  default: {}
              }
          }}


    //* calculate total TIME to run the whole program
        resetTimer(TVAL_TOTAL_2);

        long double  d = ((getTimerS(TVAL_TOTAL_2)*1000000+(getTimerMS(TVAL_TOTAL_2))) -(getTimerS(TVAL_TOTAL_1)*1000000+(getTimerMS(TVAL_TOTAL_1)) ));

        avg_time_render_grid += d/1000000;

        char str[128];
        sprintf(str, "Total time to loop the whole program (us): %0.1Lf", 
            d
           );
        renderFormattedText(renderer, str, 0 , 20);

        char str2[128];
        sprintf(str2, "Kernel: <<<%d,%d>>>, AVG_FPS(%d s): %d", gridSize, blockSize, s_cambio_threads, fps_render_grid);
        renderFormattedText(renderer, str2, 250 , 0);

        //* calculate total time to run the whole program
        resetTimer(TVAL_TOTAL_1); 
    //}

}

void langtons_ant(SDL_Renderer *renderer, state_t *state)
{
    // RENDER ANT
    SDL_SetRenderDrawColor(renderer, ANT_COLOR.r, ANT_COLOR.g, ANT_COLOR.b, 255);
    SDL_Rect ant_rect = {
        .x = state->ant.x * CELL_WIDTH ,
        .y = state->ant.y * CELL_HEIGHT,
        .w = CELL_WIDTH,
        .h = CELL_HEIGHT
    };
    SDL_RenderFillRect(renderer, &ant_rect);

    if (state->mode == RUNNING_MODE)
    for (int i = 0; i < MOVES_PER_FRAME; i++) {
        int ant_idx = (state->ant.y * N) + state->ant.y;
        int current = state->board[ant_idx];

        // TURN 90º
        switch (current) {
            case WHITE:
                state->ant.dir = mod(state->ant.dir + 1, 4);
                break;
            case BLACK:
                state->ant.dir = mod(state->ant.dir - 1, 4);
                break;
        }

        // FLIP THE COLOR OF THE SQUARE
        state->board[ant_idx] = BLACK + WHITE - current;

        // MOVE FORWARD ONE UNIT
        switch (state->ant.dir) {
            case UP:
                state->ant.y = mod(state->ant.y - 1, N);
                break;
            case RIGHT:
                state->ant.x = mod(state->ant.x + 1, N);
                break;
            case DOWN:
                state->ant.y = mod(state->ant.y + 1, N);
                break;
            case LEFT:
                state->ant.x = mod(state->ant.x - 1, N);
                break;
        }
    }
}

void game_of_life(SDL_Renderer *renderer, state_t *state)
{
    if (state->mode == RUNNING_MODE)
    for (int i = 0; i < MOVES_PER_FRAME; i++) {
        int new_board[N*N];

        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++) {
                int idx = (y * N) + x;
                int n_neigh = 
                    state->board[(mod((y - 1), N) * N) + mod((x - 1), N)] +
                    state->board[(mod((y - 1), N) * N) + mod((x    ), N)] +
                    state->board[(mod((y - 1), N) * N) + mod((x + 1), N)] +
                    state->board[(mod((y    ), N) * N) + mod((x - 1), N)] +
                    state->board[(mod((y    ), N) * N) + mod((x + 1), N)] +
                    state->board[(mod((y + 1), N) * N) + mod((x - 1), N)] +
                    state->board[(mod((y + 1), N) * N) + mod((x    ), N)] +
                    state->board[(mod((y + 1), N) * N) + mod((x + 1), N)];

                if (state->board[idx] == ALIVE && (n_neigh == 2 || n_neigh == 3))
                    new_board[idx] = ALIVE;
                else if (state->board[idx] == DEAD && n_neigh == 3)
                    new_board[idx] = ALIVE;
                else
                    new_board[idx] = DEAD;
            }

        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++){
                int idx = (y * N) + x;
                state->board[idx] = new_board[idx];
            }
    }   
}


////////////////////////////////////////////////////////////////////////////////!!
//SAND SIMULATION FUNCTIONS

//=================== GPU FUNCTIONS =======================//

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

__device__ bool sandsim_puede_moverseGPU(u_int8_t *board, u_int8_t sustancia, int idx, const int n, curandState *states)
{
    
    // Si las coordenadas se salen de los límites, no se puede mover por defecto
    if (idx < 0 || idx > (n * n) - 1) return false;
    switch (sustancia)
    {
    case SAND:
        if (board[idx] == AIR){ return true;}
        else if (board[idx] == SAND){ return false;}
        else if (board[idx] == WATER && (curand_uniform (&states[idx]) < 0.5)){ return true;}
        else if (board[idx] == ROCK){ return false;}
        else if (board[idx] == OIL){ return true;}
        else if (board[idx] == HUMO){ return true;}

        break;

    case WATER:
        if (board[idx] == AIR){ return true;}
        else if (board[idx] == SAND){ return false;}
        else if (board[idx] == WATER){ return false;}
        else if (board[idx] == ROCK){ return false;}
        else if (board[idx] == OIL && (curand_uniform (&states[idx])) < 0.5){ return true;}
        else if (board[idx] == HUMO){ return true;}
        break;

    case ROCK:
        if(board[idx] == AIR){ return true;} 
        else if (board[idx] == SAND && (curand_uniform (&states[idx])) < 0.2 ){ return true;}
        else if (board[idx] == WATER && (curand_uniform (&states[idx])) < 0.9 ){ return true;}
        else if (board[idx] == ROCK){ return false;}
        else if (board[idx] == OIL && (curand_uniform (&states[idx])) < 0.9){ return true;}
        else if (board[idx] == HUMO){ return true;}

      break;

    case FIRE:
      if(board[idx] == AIR){ return true;} 
      else if (board[idx] == SAND){ return false;}
      else if (board[idx] == WATER){ return true;}
      else if (board[idx] == ROCK){ return false;}
      else if (board[idx] == FIRE){ return false;}
      else if (board[idx] == OIL){ return true;}
      else if (board[idx] == HUMO){ return true;}
      break;

    case OIL:
      if(board[idx] == AIR){ return true;} 
      else if (board[idx] == SAND){ return false;}
      else if (board[idx] == WATER){ return false;}
      else if (board[idx] == ROCK){ return false;}
      else if (board[idx] == OIL){ return false;}
      else if (board[idx] == HUMO){ return true;}

      break;
    
    case HUMO:
      if(board[idx] == AIR){ return true;} 
      
      break;

    default:
        return false;
        break;
    }

    return false;
}

__device__ void sandsim_moverGPU(u_int8_t *board, bool *has_moved, int idxFrom, int idxTo, const int n, curandState *states)
{
    u_int8_t sustancia = board[idxFrom];
    u_int8_t otraSustancia = board[idxTo];
    //Switch para las interacciones especiales
    switch (sustancia)
    {
    case FIRE:
        if(otraSustancia == OIL){
            board[idxFrom] =FIRE;
            board[idxTo] = FIRE;
        } else if(otraSustancia == WATER){
            board[idxFrom] = HUMO;
            board[idxTo] = WATER;
        } else{
            bool seDescompone= (curand_uniform (&states[idxFrom])) < 0.003;
            if(seDescompone){
                board[idxFrom] = HUMO;
                board[idxTo] = otraSustancia;
            } else{
                board[idxFrom] = otraSustancia;
                board[idxTo] = FIRE;
            }
  
        }
        break;

    case HUMO:
        
        if (true)
        {
            bool seDescompone= (curand_uniform (&states[idxFrom])) < 0.05;
            if(seDescompone){
            board[idxFrom] = AIR;
            board[idxTo] = otraSustancia; } 
            else{
            board[idxFrom] = otraSustancia;
            board[idxTo] = sustancia;
        }}

        break;

    default:
        board[idxFrom] = otraSustancia;
        board[idxTo] = sustancia;
        break;
    }
    has_moved[idxFrom] = true;
    has_moved[idxTo] = true;
}

__device__ bool sandsim_mover_abajoGPU(u_int8_t *board, u_int8_t sustancia, bool *has_moved, int idx, const int n, curandState *states)
{
    if (sandsim_puede_moverseGPU(board, sustancia, idx + n, n, states))
    { //Mover abajo
        sandsim_moverGPU(board, has_moved, idx , idx + n, n, states);
        return true;
    }
    return false;
}

__device__ bool sandsim_mover_izq_derGPU(u_int8_t *board, u_int8_t sustancia, bool *has_moved,int idx, const int n, curandState *states){
    //random number to define if it should go left or right
    bool primeroIzquierda = (curand_uniform (&states[idx])) < 0.5;
    if(primeroIzquierda){
        if(sandsim_puede_moverseGPU(board, sustancia, idx - 1, n , states)){ //Mover a la izquierda
            sandsim_moverGPU(board, has_moved, idx, idx - 1, n, states);
            return true;
        } else if(sandsim_puede_moverseGPU(board, sustancia, idx + 1, n , states)){ //Mover a la derecha
            sandsim_moverGPU(board, has_moved, idx, idx + 1, n, states);
            return true;
        }
    } else {
        if(sandsim_puede_moverseGPU(board, sustancia, idx + 1, n , states)){ //Mover a la derecha
            sandsim_moverGPU(board, has_moved, idx, idx + 1, n, states);
            return true;
        } else if(sandsim_puede_moverseGPU(board, sustancia, idx - 1, n , states)){ //Mover a la izquierda
            sandsim_moverGPU(board, has_moved, idx, idx - 1, n, states);
            return true;
        }
    }
    return false;
}

__device__ bool sandsim_mover_abajo_diagonalGPU(u_int8_t *board, u_int8_t sustancia, bool *has_moved,int idx, const int n, curandState *states){
    //random number to define if it should go left or right
    bool primeroIzquierda = (curand_uniform (&states[idx])) < 0.5;
        if(primeroIzquierda){
            if(sandsim_puede_moverseGPU(board, sustancia, idx - 1 + n, n , states)){ //Mover abajo a la izquierda
                sandsim_moverGPU(board, has_moved, idx, idx - 1 + n, n, states);
                return true;
            } else if (sandsim_puede_moverseGPU(board, sustancia, idx + 1 + n, n , states)){ //Mover abajo a la derecha
                sandsim_moverGPU(board, has_moved, idx, idx + 1 + n, n, states);
                return true;
            }
        } else {
            if(sandsim_puede_moverseGPU(board, sustancia, idx + 1 + n, n , states)){ //Mover abajo a la derecha
                sandsim_moverGPU(board, has_moved, idx, idx + 1 + n, n, states);
                return true;
            } else if (sandsim_puede_moverseGPU(board, sustancia, idx - 1 + n, n , states)){ //Mover abajo a la izquierda
                sandsim_moverGPU(board, has_moved, idx, idx - 1 + n, n, states);
                return true;
            }
        }
    return false;
}

__device__ bool sandsim_mover_abajo_y_ladosGPU(u_int8_t *board, u_int8_t sustancia, bool *has_moved,int idx, const int n, curandState *states){
    
    //Si no se puede mover hacia abajo
    if(!sandsim_mover_abajoGPU(board, sustancia, has_moved, idx, n , states)){                 
        //Se moverá en diagonal hacia abajo
        if(sandsim_mover_abajo_diagonalGPU(board, sustancia, has_moved, idx, n , states)){
            //Si se mueve en diagonal, retornar true
            return true;
        }
    }
    return false; 
}

__device__ bool sandsim_mover_arriba_y_ladosGPU(u_int8_t *board, u_int8_t sustancia, bool *has_moved,int idx, const int n, curandState *states){
    
    if(sandsim_puede_moverseGPU(board, sustancia, idx - n, n , states)){ //Mover arriba
        sandsim_moverGPU(board, has_moved, idx, idx - n, n, states);
        return true;
    }

    bool primeroIzquierda = (curand_uniform (&states[idx])) < 0.5;

    if(primeroIzquierda){
        if(sandsim_puede_moverseGPU(board, sustancia, idx - 1 - n, n , states)){ //Mover arriba a la izquierda
            sandsim_moverGPU(board, has_moved, idx, idx - 1 - n, n, states);
            return true;
        } else if (sandsim_puede_moverseGPU(board, sustancia, idx + 1 - n, n , states)){ //Mover arriba a la derecha
            sandsim_moverGPU(board, has_moved, idx, idx + 1 - n, n, states);
            return true;
        }
    } else {
        if(sandsim_puede_moverseGPU(board, sustancia, idx + 1 - n, n , states)){ //Mover arriba a la derecha
            sandsim_moverGPU(board, has_moved, idx, idx + 1 - n, n, states);
            return true;
        } else if (sandsim_puede_moverseGPU(board, sustancia, idx - 1 - n, n , states)){ //Mover arriba a la izquierda
            sandsim_moverGPU(board, has_moved, idx, idx + 1 - n, n, states);
            return true;
        }
    }
    
    return false; 
}

__global__ void sandsimGPU(u_int8_t *board, bool *has_moved, const int n, curandState *states, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //seed diferente por cada id
    seed += idx;
    //Inicializar las funciones random de CUDA
    curand_init(seed, idx, 0, &states[idx]);

    //if (!has_moved[idx]) //Si no se ha movido en este frame
    //{
        if (board[idx] == SAND){
            sandsim_mover_abajo_y_ladosGPU(board, SAND, has_moved, idx, n, states);
        }
        if(board[idx] == ROCK){
            sandsim_mover_abajoGPU(board, ROCK, has_moved, idx, n, states);
        }

        if(board[idx] == WATER){          
            //Si el agua no se puede mover abajo o a los lados
            if(!sandsim_mover_abajo_y_ladosGPU(board, WATER, has_moved, idx, n, states)){
                //Se mueve a la izquierda o derecha
                sandsim_mover_izq_derGPU(board, WATER, has_moved, idx, n, states);;
            }   
        }
    
        if(board[idx] == OIL){
            //Si el aceite no se puede mover abajo o a los lados
            if(!sandsim_mover_abajo_y_ladosGPU(board, OIL, has_moved, idx, n, states)){
                //Se mueve a la izquierda o derecha
                sandsim_mover_izq_derGPU(board, OIL, has_moved, idx, n, states);;
            }   
        }

        if(board[idx] == FIRE){
            if (!sandsim_mover_abajo_y_ladosGPU(board, FIRE, has_moved, idx, n, states)){
                bool seDescompone= (curand_uniform (&states[idx])) < 0.2;
                if(seDescompone){
                    board[idx] = HUMO;
                } 
            }  
        }

        if(board[idx] == HUMO){
            //Si el aceite no se puede mover abajo o a los lados
            if(!sandsim_mover_arriba_y_ladosGPU(board, HUMO, has_moved, idx, n, states)){
                //Se mueve a la izquierda o derecha
                sandsim_mover_izq_derGPU(board, HUMO, has_moved, idx, n, states);;
            }   
        }

    //}
}

__global__ void resetHasMovedGPU(bool *has_moved, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    has_moved[idx] = false;
}

//=========================================================//


//Variables benchmarking Función SandSim
int sandsim_framecnt = 0;
int blocksize_sandsim = 1024;
int fps_sandsim = 0;
//double time_sandsim_acum = 0;
double avg_time_sandsim = 0;
double avg_FPS_sandsim = 0;

//***** world_sand_sim() RUNS THE SIMULATION logic for all elements of the world
void world_sand_simOnGPU(SDL_Renderer *renderer, state_t *state, u_int8_t *d_board, bool *d_has_moved, curandState *d_random, unsigned int seed)
{

    ++sandsim_framecnt; //Sumando los fps de sandsim

    if (state->mode == RUNNING_MODE){

        //CÁLCULO DE EL TIEMPO PROMEDIO PARA CADA NUMERO DE THREADS
        if(blockSize != blocksize_sandsim){
            double avg_fps = sandsim_framecnt/s_cambio_threads;
            long double avg_time = (avg_time_sandsim /avg_fps) / s_cambio_threads; 
            
            int nElem = N * N;
            int gridsize_sandsim = (nElem + blocksize_sandsim - 1) / blocksize_sandsim;

            SDL_Log("[SANDSIM] Tiempo promedio para el kernel <<<%d,%d>>> (us): %0.1Lf",gridsize_sandsim, blocksize_sandsim,avg_time);
            avg_time_sandsim = 0; //Reinciar el conteo del promedio acumulado en X segundos
            sandsim_framecnt = 0;
            blocksize_sandsim = blockSize; //Se actualiza la variable para el contador interno de sandsim
        }


        for (int i = 0; i < MOVES_PER_FRAME; i++) {
        
            //*calculate time to render the grid
            struct timeval tval_before_sandsim, tval_after_sandsim, tval_result_sandsim;
            gettimeofday(&tval_before_sandsim, NULL);

        //========== CUDA FOR SANDSIM =================//
        // set up data size of board
        int nElem = N * N;
        size_t nBytesBoards = nElem * sizeof(u_int8_t);
        size_t nBytesBool = nElem * sizeof(bool);


        // transfer data from host to device
        
        // invoke kernel at host side
        int iLen = 256;
        dim3 block(blockSize);
        dim3 grid((nElem + block.x - 1) / block.x);
        sandsimGPU<<<grid, block>>>(d_board, d_has_moved, N, d_random, seed);
        resetHasMovedGPU<<<grid, block>>>(d_has_moved, N);
        //cudaDeviceSynchronize();
        // copy kernel result back to host side
        cudaMemcpyAsync(state->board, d_board, nBytesBoards, cudaMemcpyDeviceToHost);
        //==============================================//
        
        //*calculate time to render the grid
        gettimeofday(&tval_after_sandsim, NULL);

        timersub(&tval_after_sandsim, &tval_before_sandsim, &tval_result_sandsim);

        avg_time_sandsim += tval_result_sandsim.tv_usec;

        char str[128];
        sprintf(str, "Total time to execute function world_sand_sim (us): %ld", (long int)tval_result_sandsim.tv_usec);
        renderFormattedText(renderer, str, 0 , 40);
        }
    }  
    
}
