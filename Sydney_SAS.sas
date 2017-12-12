* Final SAS project for STAT202A;
* Complete the code;

* Import car data;
proc import out= work.data
datafile= "/folders/myfolders/study1/regression_auto.csv"
dbms=csv replace; getnames=yes; datarow=2;
run;

* Compute the correlation between car length and mpg;
proc corr data = work.data;
var length mpg;
run;

* Make a scatterplot of price (x-axis) and mpg (y-axis);
proc gplot data=work.data;
plot mpg*price;
run;

* Make a box plot of mpg for foreign vs domestic cars;
proc boxplot data=work.data;
plot mpg * foreign;
run;

* Perform simple linear regression, y = mpg, x = price1; 
* Do NOT include the intercept term;
proc reg data=work.data;
model mpg=price1 /noint;
run;

* adding a squared length variable to the dataset;
data work.data2;
set work.data;
LengthSq = length*length;
run;

* Perform linear regression, y = mpg, x1 = length, x2 = length^2; 
* Include the intercept term;
proc reg data=work.data2;
model mpg=length LengthSq;
run;
