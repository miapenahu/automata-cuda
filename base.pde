final int WIDTH=256;
final int HEIGHT=256;
final int SCALE_FACTOR = 4;
final byte AIR = 0;
final byte ROCK = 1;
final byte SAND = 2;
final byte WATER = 3;
final byte OIL = 4;
final byte FIRE = 5;

byte[] world;
boolean[] seHaMovidoFlags;
int[] momentum;
 
PGraphics worldGfx;

int brushSize = 1;
boolean brushToggle = true;

void setup(){
  //size(WIDTH*SCALE_FACTOR, HEIGHT*SCALE_FACTOR, P3D);
  size(1024, 1024, P3D);
  worldGfx = createGraphics(WIDTH, HEIGHT);
  //Evitar que se suavicen los pixeles escalados
  ((PGraphicsOpenGL)g).textureSampling(2);

  world = new byte[WIDTH*HEIGHT];
  seHaMovidoFlags = new boolean[WIDTH*HEIGHT];
  momentum = new int[WIDTH*HEIGHT];
  
  //Hacer una línea de roca en la base
  for(int y = HEIGHT - 10; y < HEIGHT; ++y){
    for(int x = 0; x < WIDTH; ++x){
      world[coord(x,y)] = ROCK;
    }
  }
  
  //Agregar arena
  for(int y = 100; y < 110; ++y){
    for(int x = 100; x < 110; ++x){
      world[coord(x,y)] = SAND;
    }
  }
  
  frameRate(30);
}

void draw(){
  
  //======== Agregar elementos si se presiona el mouse ========//
  
  if(mousePressed){
      int mouseXInWorld = mouseX / SCALE_FACTOR;
      int mouseYInWorld = mouseY / SCALE_FACTOR;
      
    if(mouseButton == LEFT){  
      colocar(brushToggle ? SAND : OIL, mouseXInWorld, mouseYInWorld);
    } else if(mouseButton == CENTER){     
      colocar(ROCK, mouseXInWorld, mouseYInWorld);
    } else if(mouseButton == RIGHT){
      colocar(brushToggle ? WATER : FIRE, mouseXInWorld, mouseYInWorld);
    }
  }
  
  
  //========= ACTUALIZACIÓN DEL MUNDO ================//
  
  //Liberar las flags de los elementos movidos
  for(int y = 0; y < HEIGHT; ++y){
    for(int x = 0; x < WIDTH; ++x){
    seHaMovidoFlags[coord(x,y)] = false;
    }
  }
  
  //Mover elementos
  for(int y = HEIGHT-1; y >= 0; --y){
    for(int x = 0; x < WIDTH; ++x){
        int actualPos = coord(x,y);
        byte actualSustancia = world[actualPos];
        
        if(seHaMovidoFlags[actualPos]) continue;
        if(actualSustancia == AIR || actualSustancia == ROCK) continue;
        
        if(actualSustancia == FIRE){
            boolean fireSpread = false;
          if(puedeMoverse(FIRE, x-1, y)){
            move(x, y, x-1, y);
            world[actualPos] = FIRE;
            fireSpread = true;
          } else if(puedeMoverse(FIRE, x+1, y)){
            move(x, y, x+1, y);
            world[actualPos] = FIRE;
            fireSpread = true;
          } else if(puedeMoverse(FIRE, x, y-1)){
            move(x, y, x, y-1);
            world[actualPos] = FIRE;
            fireSpread = true;
          } else if(puedeMoverse(FIRE, x, y+1)){
            move(x, y, x, y+1);
            world[actualPos] = FIRE;
            fireSpread = true;
          }
          if(!fireSpread){
            //Si no se esparce el fuego, se quema hasta desaparecer
            world[actualPos] = AIR;
          }
          
        }
      
        if(puedeMoverse(actualSustancia ,x ,y + 1)){
          //Idealmente se quiere moverse hacia abajo
          move(x, y, x, y + 1);  
        } 
        
        //Escoger al azar si mirará primero a la izquierda o a la derecha
        boolean checkLeftFirst;
        
        //Si se tiene momentum se va a preferir que se siga moviendo en la misma dirección
        if(momentum[actualPos] == -1){
          checkLeftFirst = true;
        } else if(momentum[actualPos] == 1){
          checkLeftFirst = false;
        } else{
          //Si no hay momentum, se escoge al azar si  se mueve a izquierda o derecha
          checkLeftFirst = random(1f) < 0.5f;
        }
        
        if(checkLeftFirst){
          if (puedeMoverse(actualSustancia, x - 1, y + 1)){
            //Si no, se tratará de mover abajo a la izquierda
            move(x, y, x - 1, y + 1);
           } else if (puedeMoverse(actualSustancia, x + 1, y + 1)){
             //Y en otro caso abajo a la derecha
             move(x, y, x + 1, y + 1);
           } 
        } else{
          if (puedeMoverse(actualSustancia, x + 1, y + 1)){
            //Si no, se tratará de mover abajo a la derecha
            move(x, y, x + 1, y + 1);
          } else if (puedeMoverse(actualSustancia, x - 1, y + 1)){
            //Y en otro caso abajo a la izquierda
            move(x, y, x - 1, y + 1);
          } 
        }
        
        //Si estamos obre una capa de agua, se esparce hacia izquierda o derecha
        if((actualSustancia == WATER || actualSustancia == OIL) 
            && y < HEIGHT-1 && (world[coord(x,y+1)] == WATER || world[coord(x,y+1)] == OIL)){
          if(checkLeftFirst){
           if (puedeMoverse(actualSustancia, x - 1, y)){
            //Si no, se tratará de mover a la izquierda
            move(x, y, x - 1, y);
           } else if (puedeMoverse(actualSustancia, x + 1, y)){
             //Y en otro caso a la derecha
             move(x, y, x + 1, y);
           } 
          } else{
          if (puedeMoverse(actualSustancia, x + 1, y)){
            //Si no, se tratará de mover a la derecha
            move(x, y, x + 1, y);
          } else if (puedeMoverse(actualSustancia, x - 1, y)){
            //Y en otro caso a la izquierda
            move(x, y, x - 1, y);
          } 
        }
      }
    }
  }
  
  
  
  //============= DIBUJO DEL MUNDO ==================//
  
  worldGfx.beginDraw();
  worldGfx.loadPixels();
  for(int y = 0; y < HEIGHT; ++y){
    for(int x = 0; x < WIDTH; ++x){
      color c;
      int actualPos = coord(x,y);
      byte actualItem = world[actualPos];
      switch (actualItem){
        case AIR: c = color(0, 0, 0); break;
        case ROCK: c = color(128, 128, 128); break;
        case WATER: c = color(0, 0, 255); break;
        case SAND: c = color(255, 255, 0); break;
        case OIL: c = color(119, 0, 199); break;
        case FIRE: c = color(255, 119, 0); break;
        default: c = color(255, 0, 0); break;
      }
      
      worldGfx.pixels[actualPos] = c;
    }
  }
  
  worldGfx.updatePixels();
  worldGfx.endDraw();
  //Ahora se dibuja cada cuadro
  //Primero hay que escalarlo
  scale(SCALE_FACTOR);
  image(worldGfx, 0, 0);
}

void mouseWheel(MouseEvent event){
  if(event.getCount() < 0){
    brushSize++;
  } else{
    brushSize--;
    if(brushSize <= 0) brushSize = 1;
  }
  println("Brush size: " + brushSize);
}

void keyPressed(){
  if(key == ' '){
    brushToggle = !brushToggle; 
  }
  if(brushToggle){
    println("SAND / ROCK / WATER");  
  } else{
    println("OIL / ROCK / FIRE");  
  }
}

void colocar(byte sustancia, int xPos, int yPos){
  for(int y = max(0,yPos-brushSize); y < min(HEIGHT-1, yPos+brushSize); ++y){
    for(int x = max(0,xPos-brushSize); x < min(WIDTH-1, xPos+brushSize); ++x){
      world[coord(x,y)] = sustancia;  
    }
  }
}

void move(int fromX, int fromY, int toX, int toY){
  int fromCoord = coord(fromX, fromY);
  int toCoord = coord(toX, toY);
  byte otraSustancia = world[toCoord];
  world[toCoord] = world[fromCoord];
  world[fromCoord] = otraSustancia;
  seHaMovidoFlags[fromCoord] = true;
  seHaMovidoFlags[toCoord] = true;
  momentum[fromCoord] = 0;
  
  if(toX > fromX){ 
    momentum[toCoord] = 1;
  }else if(toX < fromX){ 
    momentum[toCoord] = -1; 
  } else {
    momentum[toCoord] = 0;
  }
}


boolean puedeMoverse(byte sustancia, int x, int y){
  //Si el valor es mayor a los límites, no estará libre
  if(x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return false;
  byte otraSustancia = world[coord(x,y)];
  if(sustancia == FIRE) return (otraSustancia == OIL);
  if(world[coord(x,y)] == AIR) return true;
  //Si la arena cayera en agua, la deja seguir cayendo
  //Pero tiene sólo la mitad de probabilidad de moverse
  if(sustancia == SAND && otraSustancia == WATER && random(1f) < 0.5f) return true;
  return false;
}

//Convertir el arreglo unidimensional world en una matriz 2D
int coord(int x, int y){
  return x + y*WIDTH;
}