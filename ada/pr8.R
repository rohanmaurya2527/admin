std<-c(40,30,20,10)
dept<-c("Science","Commerce","Arts","Management")
pie(std,labels = dept,main = "Student Distri by Dept")
barplot(std,names.arg=dept,main = "Student Distri by Dept",
        xlab="Dept",ylab="No of Students",col="lightblue")

sales<-c(40,50,60,70,80,90)
plot(sales,type="l",col="blue",
     lwd=2,main="Monthly sales",
     xlab="Months",ylab="Sales")

mks<-c(50,60,70,80,90,55,66,78,89,90)
hist(mks,col="lightgreen",main="Histogram",xlab="Marks")

n<-as.integer(readline("Enter no of data points"))
x<-numeric(n)
y<-numeric(n)
for(i in 1:n){
  x[i]<-as.numeric(readline(prompt = paste("Enter X",i,":")))
  y[i]<-as.numeric(readline(prompt = paste("Enter Y",i,":")))
}
print(x)
print(y)
plot(x,y,main="Scatter Plot",
     xlab="X val",ylab="Y val",
     pch=19,col="blue")