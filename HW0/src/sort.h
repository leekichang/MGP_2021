#ifndef __SORT_H__
#define __SORT_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

using namespace std;

class SortObject {
public:
	int n;
	int* seq;
	int* ans;
	SortObject(string i_path, string a_path);
	~SortObject();
	void Sort();
	void MakeOutputFile();
};

#endif
