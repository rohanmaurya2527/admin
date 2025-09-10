barplot(airquality$Ozone, main = 'Ozone Concentration Air', xlab = 'ozonelevels' ,horiz = FALSE)
hist(airquality$Temp, main="Airport max Temp", xlab="Temperature(Fahrenheit)",xlim=c(50,125),col="yellow",freq=TRUE)

boxplot(airquality$Wind, main="AVG Wind Speed", xlab="Miles PER Hrs", ylab="wind", col="yellow",border="red",horizontal=TRUE,notch=TRUE)

data<- matrix(rnorm(25,0,5),nrow=5,ncol=5)
colnames(data)<- paste0("col",1:5)
row.names(data)<- paste0("row",1:5)
mycolors<- colorRampPalette(c("cyan","deeppink3"))
heatmap(data,col=mycolors(100))
