#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Declare our sense and move functions
vector<double> sense(vector<double> p, string Z, vector<string> world, 
                     double pHit, double pMiss);
vector<double> move(vector<double> p, int U, double pExact, 
                    double pOvershoot, double pUndershoot);
void print_probability(vector<double> p);

int main() {
    // Declare & define all vectors and probability variables
    vector<double> p (5, 0.2);
    vector<string> world;
    vector<string> measurements;
    vector<int> motions (2, 1);
    double pHit = 0.6;
    double pMiss = 0.2;
    double pExact = 0.8;
    double pOvershoot = 0.1;
    double pUndershoot = 0.1;

    // Fill in our world and measurements vectors
    world.push_back("green");
    world.push_back("red");
    world.push_back("red");
    world.push_back("green");
    world.push_back("green");

    measurements.push_back("red");
    measurements.push_back("green");

    // Iterate through our measurements
    for (int k = 0; k < measurements.size(); k++) {
        p = sense(p, measurements[k], world, pHit, pMiss);
        p = move(p, motions[k], pExact, pOvershoot, pUndershoot);
    }

    // Print out the probability vector
    print_probability(p);

    return 0;
}

vector<double> sense(vector<double> p, string Z, vector<string> world, 
                     double pHit, double pMiss) {
    /* 
     * Updates probability based on a sensor reading.
     */

    vector<double> q (p.size());
    bool hit;
    double s = 0; // sum of the q vector's elements

    // Calculate new probabilities
    for (int i = 0; i < p.size(); i++) {
        hit = (Z == world[i]);
        q[i] = p[i] * (hit * pHit + (1-hit) * pMiss);
        s += q[i]; 
    }

    // Divide probabilities by the total sum
    for (int i = 0; i < q.size(); i++) {
        q[i] = q[i] / s;
    }

    return q;
}

vector<double> move(vector<double> p, int U, double pExact, 
                    double pOvershoot, double pUndershoot) {
    /* 
     * Updates probability based on movement.
     */

    vector<double> q (p.size());
    double s = 0; // holds probability values calculated below

    // Calculate new probabilities
    for (int i = 0; i < p.size(); i++) {
        // The modulo operator behaves a bit different in C++ with negatives
        // If we took it as -1 % p.size(), we'd get zero instead of four
        // -1 + p.size() will give us what we would get from -1 % p.size in Python
        if (i-U < 0) {
            s = pExact * p[(i-U) + p.size()];
        }
        else {
            s = pExact * p[(i-U) % p.size()];
        }
        if (i-U-1 < 0) {
            s = s + pOvershoot * p[(i-U-1) + p.size()];
        }
        else {
            s = s + pOvershoot * p[(i-U-1) % p.size()];
        }
        if (i-U+1 < 0) {
            s = s + pUndershoot * p[(i-U+1) + p.size()];
        }
        else {
            s = s + pUndershoot * p[(i-U+1) % p.size()];
        }

        q[i] = s;
    }

    return q;
}

void print_probability(vector<double> p) {
    // Iterates through the vector to print it out
    for (int i = 0; i < p.size(); i++) {
        cout << p[i] << " ";
    }
    cout << endl;
}