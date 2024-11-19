#ifndef GAUSSIAN_H
#define GAUSSIAN_H

// class declaration
class Gaussian
{
	private:
		float mu, sigma2;

	public:
		
		// constructor functions
		Gaussian ();
		Gaussian (float, float);

		// change value of average and standard deviation 
		void setMu(float);
		void setSigma2(float);

		// output value of average and standard deviation
		float getMu();
		float getSigma2();

		// functions to evaluate 
		float evaluate (float);
		Gaussian mul (Gaussian);
		Gaussian add (Gaussian);
};

#endif