#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
private:
	vector<vector<double>> means;
    vector<vector<double>> stds;
    int num_label;
    int num_obs;
    int num_vars;

public:

	vector<string> possible_labels = {"left","keep","right"};

	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);

};

#endif