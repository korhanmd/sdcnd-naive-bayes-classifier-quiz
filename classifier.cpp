#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "classifier.h"

#include <typeinfo>

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d, 
            s_dot and d_dot.
          - Example : [
                [3.5, 0.1, 5.9, -0.02],
                [8.0, -0.3, 3.0, 2.2],
                ...
            ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */

    int num_label = possible_labels.size();
    int num_obs = data.size();
    int num_vars = data[0].size();
    int label_ind;

    vector<vector<vector<double>>> totals_by_label(num_label);

    for(int i=0; i < num_label; i++){
        totals_by_label[i].resize(num_vars);
    }

    for(int i=0; i<num_obs; i++){
        if(labels[i] == possible_labels[0])
            label_ind = 0;
        else if(labels[i] == possible_labels[1])
            label_ind = 1;
        else
            label_ind = 2;
        
        for(int j=0; j < num_vars; j++){
            totals_by_label[label_ind][j].push_back(data[i][j]);
        }
    }

    cout << totals_by_label[2][0].size() << endl;

    double means[num_label][num_vars];
    double stds[num_label][num_vars];

    for(int i=0; i < num_label; i++){
        for(int j=0; j < num_vars; j++){
            int size = totals_by_label[i][j].size();
            double mean = std::accumulate(totals_by_label[i][j].begin(), totals_by_label[i][j].end(), 0.0)/size;
            means[i][j] = mean;

            std::vector<double> diff(size);
            std::transform(totals_by_label[i][j].begin(), totals_by_label[i][j].end(), diff.begin(), [mean](double x) { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / size);

            stds[i][j] = stdev;
        }
    }

    cout << stds[0][2] << endl;
}

string GNB::predict(vector<double> sample)
{
    /*
        Once trained, this method is called and expected to return 
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this
    */

    return this->possible_labels[1];

}