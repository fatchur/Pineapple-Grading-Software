#ifndef  _CSV_H_
#define _CSV_H_

using namespace std;
class CSV	{
	public:
        void readCSV(std::istream &input, std::vector< std::vector<int> > &output);
        void readCSV(std::istream &input, std::vector< std::vector<double> > &output);
        void readCSV(std::istream &input, std::vector< std::vector<float> > &output);
        void readCSV(std::istream &input, std::vector< std::vector<std::string> > &output);
        void readCSV(std::istream &input, std::vector< double> &output);
        void readCSV(std::istream &input, double &output);
        void writeCSV(std::string fileName, std::vector<float> inputChar);
        void writeCSV(std::string fileName);
};
#endif // ! _CSV_H_

