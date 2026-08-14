#Advanced Data Analysis using PivotTables and Pivot Charts in R program

#Creating a sample dataset

sales_data <- data.frame(
  
  Department = c("Science", "Commerce", "Arts", "Science", "Commerce", "Arts"),
  
  Month= c("Jan", "Jan", "Jan", "Feb", "Feb", "Feb"),
  
  Sales= c(5000, 7000, 4000, 6000, 800, 5000)
)  
#Display original dataset

print(sales_data)



#Month
# aggregate() creates a pivot table by calculating total sales department-wise

pivot_table <- aggregate(Sales ~ Department, data = sales_data, sum)

# Display pivot table

print(pivot_table)

# aggregate() calculates average sales department-wise

average_table <- aggregate(Sales ~ Department, data = sales_data, mean)

# Display average sales

print(average_table)

# table() creates frequency table

table(sales_data$Department)

# xtabs() creates cross-tabulation similar to pivot table

xtabs(Sales ~ Department + Month, data = sales_data)

# Bar Chart (Pivot Chart)

barplot(pivot_table$Sales,
        names.arg = pivot_table$Department)

# Bar Chart (Pivot Chart)

barplot(pivot_table$Sales,
        names.arg = pivot_table$Department,
        main = "Department Wise Total Sales",
        xlab = "Department",
        ylab = "Total Sales",
        col = "lightblue")