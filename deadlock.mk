DEADLOCK_ALL=deadlock

ALL+=$(DEADLOCK_ALL)
CLEAN+=$(ALL)

$(DEADLOCK_ALL): deadlock.cc
	$(CXX48) -O -g $< -o $@ -lpthread -mrtm
