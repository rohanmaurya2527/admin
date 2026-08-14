# Aim: t-Test, F-Test, ANOVA (One-Way Classification),
# Chi-Square Test, and Independence of Attributes Using R

# Code:

# Dataset 1
group1 <- c(85, 90, 78, 88, 92)

# Dataset 2
group2 <- c(80, 85, 75, 82, 88)

# t.test() compares the means of two samples
t.test(group1, group2)

# var.test() performs F-Test to compare variances
var.test(group1, group2)


# Creating data for One-Way ANOVA

group <- factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C"))

marks <- c(80, 85, 90, 88, 92, 95, 75, 78, 82)

# aov() performs One-Way ANOVA

anova_result <- aov(marks ~ group)

# Display ANOVA table

summary(anova_result)

# Creating contingency table for Chi-Square Test

data_matrix <- matrix(c(20, 30,
                        25, 25),
                      nrow = 2)

# Display contingency table

print(data_matrix)


#test of independance of atttributes

attribute_matrix <- matrix(c(30, 20,
                             25, 25),
                           nrow = 2)
#Chi -square test for independance
chisq.test(attribute_matrix)