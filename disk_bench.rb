require 'benchmark'

# N=1:
#  999997440 bytes (1.0 GB) copied, 3.29504 s, 303 MB/s
#   0.000000   0.000000   3.540000 (  3.604286)
# N=2:
#  499998720 bytes (500 MB) copied, 1.7996 s, 278 MB/s
#  499998720 bytes (500 MB) copied, 1.83317 s, 273 MB/s
#   0.000000   0.000000   3.840000 (  2.000630)


SIZE = 1000 ** 3
N = 1
CHUNK = 4096

puts Benchmark.measure{
  N.times do |i|
    fork {
      system("dd if=/dev/zero of=output#{i} bs=#{CHUNK} count=#{SIZE / N / CHUNK}")
    }
  end
  Process.waitall
}
