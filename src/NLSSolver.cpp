#include "NLSSolver.hpp"

namespace nlssolver
{

    using namespace Eigen;
    using namespace std;

    bool NLSSolver::solveByGN()
    {
        cout << "Guass-Newton Solver processing...." << endl;
        int iter = 0;

        // 误差（马氏距离）
        double current_squared_error;

        // 迭代步长的模
        double delta_norm;

        while (iter++ < maximumIterations_)
        {
            //
            computeJacobianAndError();
            current_squared_error = error_.squaredNorm();

            if (vebose_)
            {
                cout << "---Current squared error: " << current_squared_error << endl;
            }

            computeHessianAndg();
            solveLinearSystem();

            delta_norm = delta_x_.norm();
            if (delta_norm < epsilon_)
                break;

            updateStates();
        }

        cout << "Gauss-Newton Solver ends." << endl;
        cout << "Iterations: " << iter << endl;
        cout << "The optimized value of a,b,c : " << endl
             << "--- a = " << a_ << endl
             << "--- b = " << b_ << endl
             << "--- c = " << c_ << endl;
        return true;
    } // solveByGN

    // Marquardt 迭代策略
    bool NLSSolver::solveByLM()
    {
        cout << "Lenvenberg Marquardt Solver processing ..." << endl;
        int iter = 0;
        double current_squared_error = 0.0;
        double delta_norm;

        double v = 2.0;
        double rou;        // 比例因子
        double tau = 1e-5; //阻尼因子初值的缩放倍数

        double lambda;
        int inner_iterations = 10;
        int inner_iter;

        ofstream foutC("/home/divenire/0_myWorkSpace/NonLinearSolver/scripts/LM.txt"
                       // , ios::app
        );

        computeJacobianAndError();
        computeHessianAndg();

        // Initial lambda
        double maxDiagonal = 0.;
        for (int i = 0; i < 3; ++i)
        {
            maxDiagonal = max(fabs(hessian_(i, i)), maxDiagonal);
        }
        lambda = tau * maxDiagonal;

        while (iter++ < maximumIterations_)
        {
            current_squared_error = error_.squaredNorm();

            if (vebose_)
            {
                cout << "--Current squared error: " << current_squared_error << endl;
            }

            inner_iter = 0;

            // Try to find a valid step in maximum iterations : inner_iter
            while (inner_iter++ < inner_iterations)
            {

                if (vebose_)
                {
                    cout << "----Current lambda: " << lambda << endl;
                }

                Matrix3d damper = lambda * Matrix3d::Identity();
                Vector3d delta = (hessian_ + damper).inverse() * g_;
                double new_a = a_ + delta(0);
                double new_b = b_ + delta(1);
                double new_c = c_ + delta(2);
                delta_norm = delta.norm();
                // compute the new error after the step
                double new_squared_error = 0.0;
                for (size_t i = 0; i < observations_.size(); ++i)
                {
                    double error_i = evaluate(new_a, new_b, new_c, i);
                    new_squared_error += error_i * error_i;
                }

                // gain ratio
                rou = (current_squared_error - new_squared_error) /
                      (0.5 * delta.transpose() * (lambda * delta + g_) + 1e-3);

                // 输出信息
                foutC << current_squared_error << " " << g_.norm() << " " << lambda << endl;

                // a valid iteration step
                // 函数值确实下降了，接受本次迭代
                if (rou > 0)
                {
                    // update states
                    a_ = new_a;
                    b_ = new_b;
                    c_ = new_c;

                    // update lamda
                    lambda = lambda * max(1.0 / 3.0, 1 - pow((2 * rou - 1), 3));
                    v = 2;
                    break;
                }
                // An invalid iteration step
                // 近似的不对，代价函数上升了，增大阻尼因子，减小步长
                else
                {
                    // update lamda
                    lambda = lambda * v;
                    v = 2 * v;
                }
            }

            if (delta_norm < epsilon_)
                break;

            double last_error = current_squared_error;
            computeJacobianAndError();
            computeHessianAndg();
            current_squared_error = error_.squaredNorm();

            // 误差变化太小，退出
            if (abs(last_error - current_squared_error) < epsilon_)
            {
                break;
            }
        }

        cout << "LM Solver ends." << endl;
        cout << "Iterations: " << iter << endl;
        cout << "The optimized value of a,b,c : " << endl
             << "--- a = " << a_ << endl
             << "--- b = " << b_ << endl
             << "--- c = " << c_ << endl;
        return true;
    } // solveByLM

    bool NLSSolver::solveByDogLeg()
    {
        cout << "Dog Leg Solver processing ..." << endl;

        // parameters
        int iter = 0;
        double v = 2.;
        double radius = 10; // initial trust region radius
        double rou;         // gain ratio

        // The number of iterations to find a valid step
        int inner_iterations = 10;
        int inner_iter = 0;

        double current_squared_error = 0.;
        double delta_norm;

        double last_error = 0;

        ofstream foutC("/home/divenire/0_myWorkSpace/NonLinearSolver/scripts/DogLeg.txt"
                       // , ios::app
        );

        // 小于最大迭代次数
        while (iter++ < maximumIterations_)
        {
            last_error = current_squared_error;

            computeJacobianAndError();
            computeHessianAndg();

            current_squared_error = error_.squaredNorm();

            if (vebose_)
            {
                cout << "--Current squared error: " << current_squared_error << endl;
            }

            // 如果误差基本不变了
            if (abs(last_error - current_squared_error) < epsilon_)
            {
                break;
            }

            // compute step for sdd
            // 最速下降法的最优步长
            double alpha = g_.squaredNorm() /
                           (jacobian_ * g_).squaredNorm();

            // steepest descent step at current estimation point
            Vector3d h_sd = alpha * g_;

            // gauss-newton step at current estimation point
            // 高斯牛顿法的迭代步长
            Vector3d h_gn = hessian_.inverse() * g_;

            // dogleg法的迭代步长
            Vector3d h_dl;

            // iterate to find a good dog leg step
            while (inner_iter++ < inner_iterations)
            {
                // compute dog leg step for
                // current trust region radius(Delta)
                int flag_choice;
                double belta;

                // 高斯牛顿的步长在信赖域范围内，选择高斯牛顿
                if (h_gn.norm() <= radius)
                {
                    h_dl = h_gn;
                    flag_choice = 0;
                }
                // 最速下降法的步长在信赖域外，归一化到信赖域内
                else if (h_sd.norm() >= radius)
                {
                    h_dl = radius / (h_sd.norm()) * h_sd;
                    flag_choice = 1;
                }
                // 两者折中计算狗腿法步长
                else
                {
                    double c = h_sd.transpose() * (h_gn - h_sd);
                    double sqrt_temp;
                    sqrt_temp = sqrt(c * c + (h_gn - h_sd).squaredNorm() *
                                                 (radius * radius - h_sd.squaredNorm()));
                    if (c <= 0)
                    {
                        belta = (-c + sqrt_temp) / ((h_gn - h_sd).squaredNorm() + 1e-3);
                    }
                    else
                    {
                        belta = (radius * radius - h_sd.squaredNorm()) / (c + sqrt_temp);
                    }
                    h_dl = h_sd + belta * (h_gn - h_sd);
                    flag_choice = 2;
                }

                // 参数更新
                delta_norm = h_dl.norm();
                double new_a = a_ + h_dl(0);
                double new_b = b_ + h_dl(1);
                double new_c = c_ + h_dl(2);

                // compute gain ratio
                // 更新后的代价函数误差
                double new_squared_error = 0.;
                for (size_t i = 0; i < observations_.size(); ++i)
                {
                    double error_i = evaluate(new_a, new_b, new_c, i);
                    new_squared_error += error_i * error_i;
                }

                double model_decreased_error;
                // 模型的近似误差
                switch (flag_choice)
                {

                case 0:
                    model_decreased_error =
                        current_squared_error;
                    break;

                case 1:
                    model_decreased_error =
                        (radius * (2 * h_sd.norm() - radius)) / (2 * alpha);
                    break;

                case 2:
                    model_decreased_error =
                        0.5 * alpha * (1 - belta) * (1 - belta) * g_.squaredNorm() +
                        belta * (2 - belta) * current_squared_error;
                    break;
                }

                // 比例因子
                rou = (current_squared_error - new_squared_error) /
                      model_decreased_error;

                // 输出信息
                foutC << current_squared_error << " " << g_.norm() << " " << radius << endl;

                // a valid decreasing step
                // 根据比例因子来更新信赖域大小
                if (rou > 0)
                {
                    // update states
                    a_ = new_a;
                    b_ = new_b;
                    c_ = new_c;

                    // update trust region radius
                    radius = radius / max(1.0 / 3.0, 1 - pow(2 * rou - 1, 3));
                    v = 2;

                    break;
                }
                else
                {
                    radius = radius / v;
                    v = v * 2;
                }
            }

            // found
            if (delta_norm < epsilon_)
                break;
        }

        cout << "Dog Leg Solver ends." << endl;
        cout << "Iterations: " << iter << endl;
        cout << "The optimized value of a,b,c : " << endl
             << "--- a = " << a_ << endl
             << "--- b = " << b_ << endl
             << "--- c = " << c_ << endl;

        return true;
    }

} // nlssolver