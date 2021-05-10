#include "sort.h"

using namespace std;

SortObject::SortObject(string i_path, string a_path)
{
	ifstream i_file(i_path);
	ifstream a_file(a_path);
	if (i_file.is_open() && a_file.is_open())
	{
		// declare temporary string variables for file read
		string line, temp;

		// read input file
		// first line -> the length of sequence
		getline(i_file, line);
		n = stoi(line);
		seq = new int[n];
		ans = new int[n];

		// second line -> sequence
		getline(i_file, line);
		stringstream ss(line);
		int idx = 0;
		while (getline(ss, temp, ' '))
		{
			seq[idx] = stoi(temp);
			idx++;
		}

		// read answer file
		getline(a_file, line);
		ss.clear();
		ss.str(line);
		idx = 0;
		while (getline(ss, temp, ' '))
		{
			ans[idx] = stoi(temp);
			idx++;
		}

		i_file.close();
		a_file.close();
	}
	else
		throw invalid_argument("Cannot open files.");
}

SortObject::~SortObject()
{
	delete[] seq;
	delete[] ans;
}

void SortObject::Sort()
{
	int min, min_idx;
	for (int i = 0; i < n; i++)
	{
		min = seq[i];
		min_idx = i;
		for (int j = i; j < n; j++)
		{
			if (seq[j] < min)
			{
				min = seq[j];
				min_idx = j;
			}
		}
		swap(seq[i], seq[min_idx]);
	}
}

void SortObject::MakeOutputFile()
{
	ofstream o_file("result/output.txt");
	bool pass = true;

	for (int i = 0; i < n; i++)
	{
		o_file << seq[i];
		if (i != n - 1)
			o_file << " ";
		if (seq[i] != ans[i])
			pass = false;
	}

	o_file.close();

	if (pass)
		cout << "PASS!" << endl;
	else
		cout << "NON-PASS!" << endl;
}
