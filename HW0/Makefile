cxx = g++
target = sort
object = src/sort.o src/main.o

all : $(target)

$(target) : $(object)
	$(cxx) -o $@ $^

clean :
	rm $(object) $(target)
