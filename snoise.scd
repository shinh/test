(
~mod289 = {|x|
	x - floor(x * (1.0 / 289.0)) * 289.0;
};

~permute = {|x|
	~mod289.value(((x*34.0)+1.0)*x);
};

~dot2 = {|x, y|
	x[0] * y[0] + x[1] * y[1];
};

~sq = {|x|
	~dot2.value(x, x);
};

~fract = {|x|
	x - floor(x);
};

~snoise = {|v|
	var c = [
		0.211324865405187,  // (3.0-sqrt(3.0))/6.0
		0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
		-0.577350269189626,  // -1.0 + 2.0 * C.x
		0.024390243902439
	];

	var i = floor(v + v[0] * c[1] + v[1] * c[1]);
	var x0 = v - i + i[0] * c[0] + i[1] * c[0];

	var i1 = if(x0[0] > x0[1], [1.0, 0.0], [0.0, 1.0]);
	var x12 = [x0[0] + c[0] - i1[0], x0[1] + c[0] - i1[1], x0[0] + c[2], x0[1] + c[2]];

	var p, m, x, h, ox, a0, g;

	i = ~mod289.value(i);

	p = ~permute.value(~permute.value(i[1] + [0.0, i1[1], 1.0]) + i[0] + [0.0, i1[0], 1.0]);

	m = max(0.5 - [~sq.value(x0), ~sq.value([x12[0], x12[1]]), ~sq.value([x12[2], x12[3]])], 0.0);
	m = m*m ;
	m = m*m ;

	x = 2.0 * ~fract.value(p * c[3]) - 1.0;
	h = abs(x) - 0.5;
	ox = floor(x + 0.5);
	a0 = x - ox;

	m = m * (1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h));

	g = [
		a0[0] * x0[0] + h[0] * x0[1],
		a0[1] * x12[0] + h[1] * x12[1],
		a0[2] * x12[2] + h[2] * x12[3]
	];
	130.0 * ~dot2.value(m, g);
};
)

s = Server.default;
s.boot;

(
SynthDef("tone", {|freq=440|
        var out;
        out = SinOsc.ar(freq);
        out = out * Line.kr(0.5, 0, 0.5, doneAction:2);
        out = Pan2.ar(out);
        Out.ar(0, out);
    }).send(s);

~bpm = 130;
~clk = TempoClock(~bpm / 60);
~i = 0;

~clk.sched(0.0, {
	var f1 = 44000.0 * abs(~snoise.value([~i * 0.02, 0.0]));
	var f2 = 66000.0 * abs(~snoise.value([~i * 0.02, 1.0]));
	var f3 = 88000.0 * abs(~snoise.value([~i * 0.02, 2.0]));
	postln('' ++ ~i ++ ': ' ++ f1 ++ ' ' ++ f2 ++ ' ' ++ f3);
	Synth("tone", [freq: f1]);
	Synth("tone", [freq: f2]);
	Synth("tone", [freq: f3]);
	~i = (~i+1) % 10000;
	1/8;
});
)

s.quit;

//~snoise.value([2.1, 3.0]);

(
x=[
	1,2,3];
)

abs(-3);

max([1,4,5,7], 3);

[1,2,3] * [3,4,5];

fract(3.4);

[0,1,2][1];

~dot2.value([2,3], [2,3]);

~mod289.value(3.2);
~permute.value([3,5,6]);

