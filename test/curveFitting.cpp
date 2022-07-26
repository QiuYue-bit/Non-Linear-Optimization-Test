#include "NLSSolver.hpp"
#include "tic_toc.h"

#include <fstream>
#include <random>

using namespace nlssolver;
using namespace std;

class curveFitting : public NLSSolver
{
public:
    virtual inline void computeJacobianAndError()
    {
        jacobian_.resize(observations_.size(), 3);
        error_.resize(observations_.size());
        for (size_t i = 0; i < observations_.size(); ++i)
        {
            double xi = observations_[i](0);
            double yi = observations_[i](1);
            Eigen::Matrix<double, 1, 3> jacobian_i;

            // y = a*exp(-x /b) + 6* sin*(x/c)
            double model_y = a_ * exp(-xi / b_) + 6 * sin(xi / c_);

            double epsilon = 1e-8;
            double model_y_plus = a_ * exp(-xi / b_) + 6 * sin(xi / (c_ + epsilon));
            double J_num = (model_y_plus - model_y) / epsilon;
            double J_analyze = -6 * xi / (c_ * c_) * cos(xi / c_);
            // cout << "J_num:" << J_num << "J_analyze" << J_analyze << endl;

            // 计算一下小的jacobian
            jacobian_i(0, 0) = exp(-xi / b_);
            jacobian_i(0, 1) = a_ * xi / (b_ * b_) * exp(-xi / b_);
            // jacobian_i(0, 2) = J_num;
            jacobian_i(0, 2) = J_analyze;

            // 计算一下大的jacobian
            jacobian_.row(i) = jacobian_i;
            error_(i) = model_y - yi;
        }
    }

    virtual inline double evaluate(double &param_a, double &param_b, double &param_c, int i)
    {
        double xi = observations_[i](0);
        double yi = observations_[i](1);
        // exp(a*x*x - b *x + c)
        // double exp_y = a_*exp(-xi / b_) + 6* sin(xi/c_);
        double model_y = param_a * exp(-xi / param_b) + 6 * sin(xi / param_c);
        return model_y - yi;
    }



    curveFitting(double a, double b, double c) : NLSSolver(a, b, c)
    {
        a_ = a;
        b_ = b;
        c_ = c;
    }
    ~curveFitting(){};
};

// 拟合函数
// y = a*exp(-x /b) + 6* sin*(x/c)




int main(int argc, char **argv)
{
    // True states
    double a = 6, b = 20, c = 5;
    int N = 100;
    // 噪声水平
    double w_sigma = 2;
    default_random_engine generator;
    normal_distribution<double> noise(0., w_sigma);

    curveFitting solver(3.0, 3.0, 3.0);
    solver.setEstimationPrecision(1e-6);
    solver.setMaximumIterations(30);
    solver.setVerbose(0);
    // solver.information_ *= (w_sigma * w_sigma);

    


    // generate random observations
    for (int i = 0; i < N; ++i)
    {
        double x = i / 1.;
        double n = noise(generator);

        // 测量值，带噪声
        double y = a * exp(-x / b) + 6 * sin(x / c) + n;
        // double y = a * x * x + b * x + c + n;
        solver.addObservation(x, y);
    }

    // 设置信息矩阵
    solver.information_.resize(solver.observations_.size(),solver.observations_.size());
    solver.information_.setIdentity();
    solver.information_ *= (1/(w_sigma * w_sigma));


    TicToc t_solve;
    // solve by gauss-newton
    solver.setInitialStates(10., 50., 6.5);
    solver.solveByGN();
    cout << "GN time:" << t_solve.toc() << "ms" << endl;
    cout << "============================================================================" << endl;

    // solve by LM
    t_solve.tic();
    solver.setInitialStates(10., 50., 6.5);
    solver.solveByLM();
    cout << "LM time:" << t_solve.toc() << "ms" << endl;
    cout << "============================================================================" << endl;

    // solve by dogleg
    t_solve.tic();
    solver.setInitialStates(10., 50., 6.5);
    solver.solveByDogLeg();
    cout << "dogleg time:" << t_solve.toc() << "ms" << endl;

    // 存储数据到文本文件
    solver.dataLogging();


    return 0;
}