# HT Test
run: HTtest
	./HTtest 5000 64 500000 10 0
	./HTtest 5000 64 500000 10 1

HTtest: main.cc 
	g++ -std=c++11 -g -pthread  $< -o $@ -lboost_system  -lboost_thread

clean:
	rm -f HTtest *.err *.log *.out

# Condor
remote_base: HTtest
	condor_submit HW2_base.cmd

remote_better: HTtest
	condor_submit HW2_better.cmd

report:
	cat base.out
	cat better.out

queue:
	condor_q

status:
	condor_status

remove:
	condor_rm

# Development
format:
	clang-format -i -style=Google *.cc *.h
