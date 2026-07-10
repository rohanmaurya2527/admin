#create date
d1<-as.Date("2026-07-05")
#Display current sys date
today<-Sys.Date()
#Format
format(d1,"%d-%m-%Y")
format(d1,"%B %d, %Y")
#Date Arithmetic
fd<-d1+10
pd<-d1-5
#Create another date
d2<-as.Date("2026-07-20")
#Diff bet 2 Dates
date_diff<-d2-d1
#Extract date comp
year<-format(d1,"%Y")
month<-format(d1,"%m")
day<-format(d1,"%d")
weekday<-format(d1,"%A")
#Display
print(d1)
print(today)
print(fd)
print(pd)
print(date_diff)
print(year)
print(month)
print(day)
print(weekday)
