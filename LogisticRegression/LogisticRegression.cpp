#include "LogisticRegression.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <list>

Eigen::MatrixXd LogisticRegression::Sigmoid(Eigen::MatrixXd Z)
{
    return 1 / (1 + (-Z.array()).exp());
}
