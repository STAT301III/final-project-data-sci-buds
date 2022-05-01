#eda 

#load packages 
library(tidyverse)
library(tidymodels)
library(janitor)
library(naniar)
library(ggplot2)

#read in 
superstore_dat <- read_csv(file = "data/unprocessed/Superstore_data.csv") %>% 
  clean_names()

# initial split 
#split data
superstore_split <- 
  initial_split(superstore_dat,
                prop = .8,
                strata = profit)

superstore_train <- training(superstore_split)
superstore_test <- testing(superstore_split)

#folds
superstore_resamples <- 
  superstore_train %>% 
  vfold_cv(v = 5, repeats = 3, strata = profit)


# initial overview of data ----

#9994 observations and 21 features 
dim(superstore_dat)

#check for missingness 
miss_var_table <- miss_var_table(superstore_train)
miss_var_summary <- miss_var_summary(superstore_train)
gg_miss_var <- gg_miss_var(superstore_train)

#no missingness in the data 
miss_var_table
miss_var_summary
gg_miss_var

# essential findings  ----

# response variables ----
# univariate sales graph
ggplot(superstore_dat, aes(sales)) +
  geom_density() +
  xlim(c(0,1000)) +
  ggtitle("Sales of Products")

# univariate profit graph
ggplot(superstore_dat, aes(profit)) +
  geom_density() +
  xlim(c(-200, 300)) +
  ggtitle("Profit or Loss Incurred from Sales")

# important predictor variables

# univariate category graph
ggplot(superstore_dat, aes(category)) +
  geom_bar() +
  ggtitle("Category of Products Sold")

# univariate subcategory graph
ggplot(superstore_dat, aes(sub_category)) +
  geom_bar() +
  ggtitle("Subcategory of Products Sold") +
  coord_flip()

#univariate region graph
ggplot(superstore_dat, aes(region)) +
  geom_bar() +
  ggtitle("Region where Customer is From")

# univariate ship mode graph
ggplot(superstore_dat, aes(ship_mode)) +
  geom_bar() +
  ggtitle("Order Ship Mode")

# univariate discount graph
ggplot(superstore_dat, aes(discount)) +
  geom_density() +
  ggtitle("Discount Provided")

# univariate state graph
ggplot(superstore_dat, aes(state)) +
  geom_bar() +
  ggtitle("Customer State of Residence") +
  coord_flip()

#univariate quantity graph
ggplot(superstore_dat, aes(quantity)) +
  geom_density() +
  ggtitle("Quantity of the Product Ordered") 

# secondary findings ----
