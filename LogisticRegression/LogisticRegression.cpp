#include "LogisticRegression.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <list>

Eigen::MatrixXd LogisticRegression::Sigmoid(Eigen::MatrixXd Z)
{
    return 1 / (1 + (-Z.array()).exp());
}

std::tuple<Eigen::MatrixXd, double, double> LogisticRegression::Propagate(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y, double lambda)
{
    int m = y.rows();
    Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array() + b;
    Eigen::MatrixXd A = Sigmoid(Z);

    auto cross_entropy = -(y.transpose() * (Eigen::VectorXd)A.array().log().transpose() +
                           ((Eigen::VectorXd)(1 - y.array())).transpose() * (Eigen::VectorXd)(1 - A.array()).log().transpose()) /
                         m;
    auto l2_reg_cost = W.array().pow(2).sum() * (lambda / (2 * m));

    double cost = static_cast<const double>((cross_entropy.array()[0])) + l2_reg_cost;

    Eigen::MatrixXd dw = (Eigen::MatrixXd)(((Eigen::MatrixXd)(A - y.transpose()) * X) / m) + ((Eigen::MatrixXd)(lambda / m * W)).transpose();

    double db = (A - y.transpose()).array().sum() / m;

    return std::make_tuple(dw, db, cost);
}

std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> LogisticRegression::Optimize(Eigen::MatrixXd W,
                                                                                                             double b,
                                                                                                             Eigen::MatrixXd X,
                                                                                                             Eigen::MatrixXd y,
                                                                                                             int num_iter,
                                                                                                             double learnin_rate,
                                                                                                             double lamba,
                                                                                                             bool log_cost)
{
    std::list<double> costsList;

    Eigen::MatrixXd dw;
    double db, cost;

    for (int i = 0; i < num_iter; i++)
    {
        std::tuple<Eigen::MatrixXd, double, double> propagate = Propagate(W, b, X, y, lamba);
        std::tie(dw, db, cost) = propagate;

        W = W - (learnin_rate * dw).transpose();
        b = b - (learnin_rate * db);

        if (1 % 100 == 0)
        {
            costsList.push_back(cost);
        }

        if (log_cost && 1 % 100 == 0)
        {
            std::cout << "Cost after iteration " << i << ": " << cost << std::endl;
        }
    }

    return std::make_tuple(W, b, dw, db, costsList);
}