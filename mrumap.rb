WANT_LRU = false

class MRUMap
  class Node
    attr_accessor :value, :time, :prv, :nxt

    def initialize(value)
      @value = value
    end

    def to_s
      "(#@value@#@time)"
    end
  end

  def initialize(capacity)
    # value => Node.
    @m = {}
    @head = nil
    @median = nil
    @last = nil
    @num_new = 0
    @num_old = 0
    @time = -1
    @capacity = capacity
  end

  def add(value)
    @time += 1

    if @m[value]
      node = @m[value]
      if node == @median
        @median = node.nxt
        @num_old -= 1
      elsif node.time < @median.time
        @num_old -= 1
      else
        @num_new -= 1
      end
      puts "UPDATE #{node}"
      if node.prv
        node.prv.nxt = node.nxt
      else
        @head = node.nxt
      end
      if node.nxt
        node.nxt.prv = node.prv
      else
        @last = node.prv
      end
      node.time = @time
    else
      node = Node.new(value)
      @m[value] = node
      node.time = @time
      puts "ADD #{node}"
    end
    node.prv = node.nxt = nil

    if @head
      @head.prv = node
      node.nxt = @head
      @head = node
      @num_new += 1

      if @num_new + @num_old + 1 > @capacity
        if WANT_LRU
          drop_node = @last
          @last = drop_node.prv
          @last.nxt = nil
        else
          drop_node = @median
          drop_node.prv.nxt = drop_node.nxt
          drop_node.nxt.prv = drop_node.prv
          @median = drop_node.nxt
        end

        puts "DROP #{drop_node}"
        @num_old -= 1
        raise "broken" if !@m.delete(drop_node.value)
      end

      while @num_new > @num_old
        @num_new -= 1
        @num_old += 1
        @median = @median.prv
      end

    else
      @head = @median = @last = node
    end

  end

  def to_s
    n = @head
    a = []
    while n
      s = ''
      s += 'H' if n == @head
      s += 'M' if n == @median
      s += 'L' if n == @last
      s += n.to_s
      a << s
      n = n.nxt
    end
    a * ' ' + " new=#{@num_new} old=#{@num_old}"
  end
end

NUM_TESTS = 1000
CAPACITY = 20
VALUES = [*1..99]

mru = MRUMap.new(CAPACITY)
NUM_TESTS.times do
  mru.add(VALUES[rand(VALUES.size)])
  puts mru
end

