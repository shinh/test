

for i in 0..10
  class C
    def f
      puts 'hello'
    end

    break

    def g
      puts 'world'
    end
  end
end

C.new.f
C.new.g
