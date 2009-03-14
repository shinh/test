d=Hash.new [];scan(/\w+/){d[$&]+=[$.]}while gets;p d
#d={};scan(/\w+/){d[$&]=(d[$&]||[])+[$.]}while gets;p d
#d={};scan(/\w+/){d[$&]||=[];d[$&]<<$.}while gets;p d
#d=Hash.new [];$<.map{|l|l.scan(/\w+/){d[$&]+=[$.]}};p d
