/*
solve y=exp(ax^2+bx+c)
by gauss-newton, LM and Dog-Leg algorithm
*/

#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <Eigen/StdVector>

using namespace std;

namespace nlssolver
{

    class NLSSolver
    {

    public:
        NLSSolver(double a, double b, double c)
            : a_(a), b_(b), c_(c) {}

        // Solve system using Gauss-Newton algorithm
        bool solveByGN();
        // Solve system using LenvenbergMarquardt algorithm
        bool solveByLM();
        // Solve system using Dog-Leg algorithm
        bool solveByDogLeg();

        // add observation
        virtual void addObservation(double x, double y)
        {
            observations_.push_back(Eigen::Vector2d(x, y));
        }

        // set initial states
        virtual void setInitialStates(double a, double b, double c)
        {
            a_ = a;
            b_ = b;
            c_ = c;
        }

        virtual void dataLogging()
        {
            ofstream foutC("/home/divenire/0_myWorkSpace/NonLinearSolver/scripts/CurveFitting_data.txt"
                           // , ios::app
            );

            //  ========================== 原始观测数据 ========================

            foutC.setf(ios::fixed, ios::floatfield);
            foutC.precision(5);

            auto covar_p = hessian_.inverse();
            Eigen::Vector3d sigma_p(sqrt(covar_p(0, 0)),
                                    sqrt(covar_p(1, 1)),
                                    sqrt(covar_p(2, 2)));

            cout << "sigma_p:" << endl
                 << "a:" << sigma_p(0) << endl
                 << "b:" << sigma_p(1) << endl
                 << "c:" << sigma_p(2) << endl;

            Eigen::VectorXd sigma_y;
            sigma_y.resize(observations_.size());

            for (size_t i = 0; i < observations_.size(); ++i)
            {
                auto jacobian_i = jacobian_.row(i);

                sigma_y(i) = jacobian_i * covar_p * jacobian_i.transpose();
                sigma_y(i) = sqrt(sigma_y(i));
                // x y sigma_y
                foutC << observations_[i](0) << " " << observations_[i](1) << " " << sigma_y(i) << " " << endl;

                // cout << "sigma_y" << i << ":" << sigma_y(i) << endl;
            }
            cout << "sigma_y1:" << sigma_y(0) << endl;
        }

        // Set maximum numbers of iteration
        void setMaximumIterations(int num)
        {
            maximumIterations_ = num;
        }
        // Set estimation precision
        void setEstimationPrecision(double epsilon)
        {
            epsilon_ = epsilon;
        }

        // Compute jacobian matrix
        virtual inline void computeJacobianAndError()
        {
            jacobian_.resize(observations_.size(), 3);
            error_.resize(observations_.size());

            for (size_t i = 0; i < observations_.size(); ++i)
            {
                double xi = observations_[i](0);
                double yi = observations_[i](1);

                Eigen::Matrix<double, 1, 3> jacobian_i;
                double exp_y = exp(a_ * xi * xi + b_ * xi + c_);

                // 计算一下单个残差的jacobian
                jacobian_i(0, 0) = exp_y * xi * xi;
                jacobian_i(0, 1) = exp_y * xi;
                jacobian_i(0, 2) = exp_y;
                // 计算一下大的jacobian
                jacobian_.row(i) = jacobian_i;
                error_(i) = exp_y - yi;
            }
        }

        virtual inline double evaluate(double &param_a, double &param_b, double &param_c, int i)
        {
            double xi = observations_[i](0);
            double yi = observations_[i](1);
            double exp_y = exp(param_a * xi * xi + param_b * xi + param_c);
            return exp_y - yi;
        }

        // Compute hessian matrix and b
        // Hessian = Jt J   g = -Jt e
        inline void computeHessianAndg()
        {
            hessian_ = jacobian_.transpose() * information_ * jacobian_;
            g_ = -jacobian_.transpose() * information_ * error_;
            // hessian_ = jacobian_.transpose() * jacobian_;
            // g_ = -jacobian_.transpose() * error_;
        }

        // Solve linear system Hx = g
        // 直接求逆
        inline void solveLinearSystem()
        {
            delta_x_ = hessian_.inverse() * g_;
        }

        // Update states
        inline void updateStates()
        {
            a_ += delta_x_(0);
            b_ += delta_x_(1);
            c_ += delta_x_(2);
        }

        inline void setVerbose(bool flag)
        {
            vebose_ = flag;
        }

        // The states that need to be estimated
        double a_;
        double b_;
        double c_;

        bool vebose_;

        int maximumIterations_;
        double epsilon_;

        // Jacobian matrix
        Eigen::MatrixXd jacobian_;

        // Hessian matrix
        Eigen::Matrix3d hessian_;
        Eigen::Vector3d g_;

        // estimation error
        Eigen::VectorXd error_;

        // squared estimation error
        double squaredError_;

        // information
        Eigen::MatrixXd information_;

        // delta x
        Eigen::Vector3d delta_x_;

        // observations
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> observations_;

    }; // class NLSSolver

} // namespace nlssolver