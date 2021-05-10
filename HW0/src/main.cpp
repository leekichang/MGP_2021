#include "sort.h"

int main(int argc, char** argv) {
	SortObject so(argv[1], argv[2]);
	
	so.Sort();
	so.MakeOutputFile();

	return 0;
}
