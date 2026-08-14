vdata<-c(10,20,30,40,50)
print(vdata)
sdata <- data.frame(
  Name=c('Amit','Priya','Rahul'),
  Age=c(20,30,14),
  Marks=c(85,90,78)
)
print(vdata[2])
print(sdata$Name)
print(sdata$Marks)

list_data <- list( 
  name = 'Rahul', 
  age = 22, 
  percentage = 85.5, 
  passed = TRUE 
)
print(list_data)