# Celullar Automata
Implementation of different celullar automata with SDL2 in C. Developed and tested on Linux, I have no idea how to (if it can be done at all) run it on other operating systems.

## Dependencies
* GCC
* GNU Make
* SDL2

## Installation
* `git clone https://github.com/joaquin-rossi/celullar-automata`
* `cd celullar-automata`
* nvcc \`sdl2-config --cflags --libs\`  src/main.cu src/render.cu src/util.cu -o automata-cuda -lSDL2 -lSDL2_ttf
* `./bin/main`

## Profiling
*`sudo nvprof ./automata-cuda sandsim`

### With api trace and log file
*`sudo nvprof --print-api-trace --log-file [LOG]sandsim-api-trace.log ./automata-cuda sandsim`

## Command

nvcc `sdl2-config --cflags --libs`  src/main.cu src/render.cu src/util.cu -o automata-cuda -lSDL2 -lSDL2_ttf

## Available automata
* [Langton's ant](https://en.wikipedia.org/wiki/Langton%27s_ant)
* [Conway's game of life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
* [Brian's brain](https://en.wikipedia.org/wiki/Brian%27s_Brain)
* [Wireworld](https://en.wikipedia.org/wiki/Wireworld)

## Extra info
* Press spacebar to toggle between paused and unpaused.
* Press left click on a cell to alter its value.
* Edit src/logic.h (pre-processor #defines) to change speed, size, etc.

## Demo
![Langton's ant](demo.gif)
