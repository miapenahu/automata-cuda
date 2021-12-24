#ifndef LOGIC_H_
#define LOGIC_H_

//Max tested value: 1200
#define N 400
#define SCREEN_WIDTH 2*N
#define SCREEN_HEIGHT 2*N
#define CELL_WIDTH (SCREEN_WIDTH / N)
#define CELL_HEIGHT (SCREEN_HEIGHT / N)
#define MOVES_PER_FRAME 1
#define MOVES_PER_SECOND 60

//number of threads maximum  we use to make the efficiency curves
#define THREADS 16

//* this two defines are to use the sys/time.h include , so they must be declared before, we can also declare them in the command line when running the code with -D_DEFAULT_SOURCE  -D_BSD_SOURCE
//#define _DEFAULT_SOURCE
//#define _BSD_SOURCE

enum AUTOMATA {
    LANGTONS_ANT,
    GAME_OF_LIFE,
    BRIANS_BRAIN,
    WIREWORLD,
    FALLING_SAND_SIM,
};

enum MODE {
    RUNNING_MODE,
    PAUSED_MODE,
    QUIT_MODE
};

enum CELL {
    BLACK,//0
    WHITE,
    BLUE,
    RED,
    GRAY,
    YELLOW,
    PURPLE,
    GREEN,
    WHITEBLUE, // 176, 241, 247
    GRAYSMOKE,
    STRUCTURE
};

/* CELL NAME MACROS */

// langtons un solo punto se mueve de forma aleatoria y va creando estructuras

// The Game of Life se mueve automatico toda la estructura de forma aleatoria
#define DEAD BLACK
#define ALIVE WHITE


// Falling sand simulator - materiales básicos
#define AIR BLACK
#define ROCK GRAY
#define SAND YELLOW
#define WATER BLUE
#define OIL PURPLE
#define FIRE RED
#define CLOUD WHITEBLUE
#define HUMO GRAYSMOKE
#define ESTATICO STRUCTURE 


typedef struct {
    int x;
    int y;
    int dir;
} position;

typedef struct {
    u_int8_t board[N*N];
    int mode;
    position ant;
} state_t;

enum ORIENTATION {
    UP,
    RIGHT,
    DOWN,
    LEFT
};

#endif // LOGIC_H_
