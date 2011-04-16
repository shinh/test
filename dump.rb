#!/usr/bin/env ruby

class Sandbox
  attr_reader :resolved

  def initialize(binding)
    @binding = binding
    @resolved = {}
  end

  def method_missing(method)
    m = method.to_s
    result = eval(m, @binding)
    resolved[m] = result
    result
  end
end

class Binding
  def dump(expr)
    sandbox = Sandbox.new(self)

    result = sandbox.instance_eval(expr)
    sandbox.resolved.each do |name, value|
      expr.gsub!(/\b#{name}\b/){"#{name}(#{value})"}
    end
    puts "#{result} = #{expr}"
  end
end

if __FILE__ == $0
  a = 42
  b = 90
  c = 20
  binding.dump "(a + b).to_f / c"
end
