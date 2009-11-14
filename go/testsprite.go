package main

import "fmt"
import "rand"
import "sdl"

const NUM_SPRITES = 100

const W = 640 - 32;
const H = 480 - 32;

func main() {
	sdl.Init(sdl.INIT_VIDEO);

	screen := sdl.SetVideoMode(640, 480, 32, 0);
	image := sdl.Load("../icon.bmp");
	image.SetColorKey(sdl.RLEACCEL | sdl.SRCCOLORKEY, image.MapRGB(255, 255, 255));

	x := make([]int16, NUM_SPRITES);
	y := make([]int16, NUM_SPRITES);
	vx := make([]int16, NUM_SPRITES);
	vy := make([]int16, NUM_SPRITES);

	rg := rand.New(rand.NewSource(0));

	for i := 0; i < NUM_SPRITES; i++ {
		//x[i] = rg.Intn(screen.w - image.w);
		//y[i] = rg.Intn(screen.h - image.h);
		x[i] = int16(rg.Intn(W));
		y[i] = int16(rg.Intn(H));
		vx[i] = 0;
		vy[i] = 0;
		for vx[i] == 0 && vy[i] == 0 {
			vx[i] = int16(rg.Intn(3) - 1);
			vy[i] = int16(rg.Intn(3) - 1);
		}
	}

	t := 0;
	running := true;
	start := sdl.GetTicks();
	//rect := sdl.Rect{ 0, 0, 0, 0 };
	//rectp := &rect;

	for running {
		t++;

		e := &sdl.Event{};

		for e.Poll() {
			switch e.Type {
			case sdl.QUIT:
				running = false;
			case sdl.KEYDOWN:
				if e.Keyboard().Keysym.Sym == sdl.K_ESCAPE {
					running = false;
				}
			}
		}

		screen.FillRect(nil, 0);

		for i := 0; i < NUM_SPRITES; i++ {
			x[i] += vx[i];
			y[i] += vy[i];
			if x[i] < 0 || x[i] >= W {
				vx[i] = -vx[i];
				x[i] += vx[i];
			}
			if y[i] < 0 || y[i] >= H {
				vy[i] = -vy[i];
				y[i] += vy[i];
			}

			//rect.X = x[i];
			//rect.Y = y[i];
			//screen.Blit(rectp, image, nil);
			screen.Blit(&sdl.Rect{ x[i], y[i], 0, 0 }, image, nil);
		}

		screen.Flip();
	}

	elapsed := sdl.GetTicks() - start;
	fmt.Printf("%f\n", float(t) / float(elapsed) * 1000.0);

	image.Free();
	screen.Free();

	sdl.Quit();
}
