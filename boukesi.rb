@m = {}

def search(board, depth=0)
  is_player = depth % 2 == 0

  if board.empty?
    return is_player
  end

#  p board

  if @m[board]
    return !(@m[board] ^ is_player)
  end

  board.each_with_index do |v, i|
    v.times do |s|
      1.upto(v) do |l|
        if s + l > v
          next
        end

        b = board.dup
        b.delete_at(i)
        b << s if s > 0
        rl = v - s - l
        b << rl if rl > 0

        r = search(b, depth+1)
        if r && depth == 0
          puts "#{i+1} #{s}-#{l}"
        end

        if r == is_player
          @m[board] = true
          return is_player
        end
      end
    end
  end

  @m[board] = false
  return !is_player
end

#board = [1,2,3,4,5]
#search(board)

#board = [2,1,4,5]
#search(board)

#board = [1,4,3]
#search(board)

board = [1,2,1]
search(board)

