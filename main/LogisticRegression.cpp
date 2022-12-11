#include "../ETL/ETL.h"

#include <vector>

int main(int argc, char *argv[])
{
    ETL etl(argv[1], argv[2], argv[3]);

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    std::cout << "Rows " << rows << std::endl;
    std::cout << "Cols " << cols << std::endl;

    return EXIT_SUCCESS;
}