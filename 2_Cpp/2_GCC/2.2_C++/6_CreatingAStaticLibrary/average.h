

class Average{
private:
    int count;
    double total;
public:
    Average(void){
        count = 0;
        total = 0.0;
    }
    void insertValue(double value);
    int getCount(void);
    double getTotal(void);
    double getAverage(void);
};