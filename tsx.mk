TSX_ALL=tsx tsx_abort
TSX_OBJS=$(TSX_ALL:=.o)
TSX_ASMS=$(TSX_OBJS:.o=.s)
TSX_SRCS=$(TSX_OBJS:.s=.cc)

ALL+=$(TSX_ALL)
CLEAN+=$(ALL) $(TSX_ASMS) $(TSX_OBJS)

CXX48=g++-4.8
CXX48=/usr/local/stow/gcc-git/bin/g++
AS=/usr/local/stow/binutils-git/bin/as

$(TSX_ALL): tsx%: tsx%.o
	$(CXX48) -O -g -std=c++11 -lpthread -fgnu-tm -march=corei7 -mtune=corei7 -mavx2 -mrtm $< -o $@

$(TSX_OBJS): tsx%.o: tsx%.s
	$(AS) $< -o $@

$(TSX_ASMS): tsx%.s: tsx%.cc
	$(CXX48) -S -O -g -std=c++11 -lpthread -fgnu-tm -march=corei7 -mtune=corei7 -mavx2 -mrtm $< -o $@
