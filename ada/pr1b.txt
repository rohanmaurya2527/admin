student<-data.frame(
  Name=c('Amit','Priya','Rahul'),
  Age=c(20,21,22),
  Marks=c(85,90,78)
)
show(student)
student$Marks
student[,c("Name","Marks")]
student[student$Marks>80]
student[2,]
student$Grade<-c("A","A+","B")
student<-rbind(student,list(Name="Suraj",Age=24,Marks=68,Grade="B"))
student$Marks<-student$Marks+5
student[order(student$Marks), ]
student[order(-student$Marks), ]
names(student)[3]<-"Score"

student$Age<- NULL
summary(student)

mean(student$Score)
max(student$Score)
min(student$Score)
