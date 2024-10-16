

# loading the required packages
library(shiny)
library(shinyFiles)
library(ggplot2) #For data visualization
library(dplyr)  # For data manipulation
library(tidyverse)
#install.packages('aws.s3')
library(aws.s3) #to access the aws s3 package
library(caret) # Load the caret package
library(pROC) # for the ROC curve
library(class) #calling the knn function
library(e1071)  # for naiveBayes function

# Set AWS credentials
Sys.setenv("AWS_ACCESS_KEY_ID" = "AKIASXWFQWBV2AEG7EEN",
           "AWS_SECRET_ACCESS_KEY" = "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",
           "AWS_DEFAULT_REGION" = "us-east-2")

# Define UI
ui <- fluidPage(
  
  titlePanel("DS 6306 FINAL PROJECT"),
  
  # Running the Regression Prediction Model
  sidebarLayout(
    sidebarPanel(
      # Requesting the user to select prediction variables
      checkboxGroupInput("variables1", "Select Variables for the Prediction Regression Model:",
                         choices = c("JobLevel", "JobRole", "TotalWorkingYears"),
                         selected = c("JobLevel", "JobRole", "TotalWorkingYears")),
      actionButton("runRegressionModel", "Run Regression Model")
    ),
    
    mainPanel(
      textOutput("rmseOutput")
    )
  ), 
  
  # Running  the classification prediction model
  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput("variables2", "Select Variables for the Prediction Regression Model:",
                         choices = c("TotalWorkingYears", "StockOptionLebvel", "JobInvolvement"),
                         selected = c("TotalWorkingYears", "StockOptionLebvel", "JobInvolvement")),
      actionButton("runClassificationModel", "Run Classification Model")
    ),
    
    mainPanel(
      textOutput("reply")
    )
  )
)


# Define server logic to plot various variables against mpg ----
server <- function(input, output, session) {
  
  #Running the Regression model
  observeEvent(input$runRegressionModel, {
    # Load data from S3
    bucket <- "msdsds6306"
    file_path <- "CaseStudy2-data.csv" # Specify the path to your CSV file in the S3 bucket

    data <- aws.s3::get_object(object = file_path, bucket = bucket)
    data <- read.csv(text = rawToChar(data))
    
    #setting the seed to have a constant resullt
    set.seed(123)
    
    # Subset data
    trainIndices <- sample(seq_len(nrow(data)), round(0.7 * nrow(data)))
    train <- data[trainIndices, ]
    test <- data[-trainIndices, ]
    
    # Fit the linear regression model
    model <- lm(MonthlyIncome ~ ., data = train[, c("MonthlyIncome", input$variables1)])
    # Make predictions
    predictions <- predict(model, newdata = test)
    
    # Calculate residuals
    residuals <- predictions - test$MonthlyIncome
    
    # Compute RMSE
    rmse <- sqrt(mean(residuals^2))
    
    # Output RMSE
    output$rmseOutput <- renderText({
      paste("Root Mean Squared Error (RMSE):", round(rmse, 2))
    })
  })
  
  #Running the Classification
  observeEvent(input$runClassificationModel, {
    # Load data from S3
    bucket <- "msdsds6306"
    file_path <- "CaseStudy2-data.csv" # Specify the path to your CSV file in the S3 bucket
    
    Talent_Train <- aws.s3::get_object(object = file_path, bucket = bucket)
    Talent_Train <- read.csv(text = rawToChar(Talent_Train))
    
    #setting the seed to have a constant resullt
    set.seed(123)
    
    # Subset data
    Talent_Clean <- Talent_Train %>% select(Attrition, TotalWorkingYears, StockOptionLevel, JobInvolvement) #selecting our variables
    trainIndices <- sample(seq(1:length(Talent_Clean$Attrition)), round(0.7 * length(Talent_Clean$Attrition)))
    train <- Talent_Clean[trainIndices, ]
    test <- Talent_Clean[-trainIndices, ]
    
    train$Attrition <-factor(train$Attrition, levels=c('Yes','No')) #factoring the response variable
    test$Attrition <-factor(test$Attrition, levels=c('Yes','No')) #factoring the response variable
    
    # Sample only a subset of 'No' instances to balance the classes
    OnlyNoAttrition <- train %>% filter(Attrition == "No")
    numYesAttrition <- sum(train$Attrition == "Yes")
    sampled_NoAttrition <- OnlyNoAttrition[sample(seq(1, nrow(OnlyNoAttrition), 1), numYesAttrition),]
    
    # Combine sampled 'No' instances with 'Yes' instances to create a balanced dataset
    balanced_data <- rbind(train %>% filter(Attrition == "Yes"), sampled_NoAttrition)
    
    #dim(balanced_data)
    
    classifications = knn(balanced_data[,2:4],train[,2:4], balanced_data[,1], prob = TRUE, k = 5) # using the F (original dataset) as the test set
    
    table(classifications,train[,1])
    CM = confusionMatrix(table(classifications,train[,1]), mode = "everything")
    #CM
    
    #Get Macro F1
    #print("test 1")
    #print(train$Attrition)
    train$Attrition = relevel(train$Attrition, ref = 'Yes')
    classifications = knn(train[,2:4],train[2:4],train[,1], prob = TRUE, k = 5)
    CM_Yes = confusionMatrix(table(classifications,train[,1]), mode = "everything") # Note F1
    
    #CM_Yes
    #print("test 2")
    train$Attrition = relevel(train$Attrition, ref = 'No')
    classifications = knn(train[,2:4],train[2:4],train[,1], prob = TRUE, k = 5)
    CM_No = confusionMatrix(table(classifications,train[,1]), mode = "everything") # Note F1
    
    #CM_No
    
    Macro_F1_Under = mean(c(CM_Yes[4]$byClass["F1"],CM_No[4]$byClass["F1"])) 
    #Macro_F1_Under
    
    test$Attrition = relevel(test$Attrition, ref = 'Yes')
    
    knn_model <- knn(balanced_data[,2:4], test[, c(2,3,4)], balanced_data[,1], prob = TRUE,  k = 5)
    
    CM = confusionMatrix(table(knn_model,test[,1]), mode = "everything")
    #CM
    
    # Evaluate model performance
    
    accuracy_knn <- round(CM$overall["Accuracy"], 4) #limiting to 4 dp for better display
    sensitivity_knn <- round(CM$byClass["Sensitivity"], 4) #limiting to 4 dp for better display
    specificity_knn <- round(CM$byClass["Specificity"], 4) #limiting to 4 dp for better display
    
    # Output sensitivity and specificity
    output$reply <- renderText({
      paste("Here are the metrics.Sensitivity: ", sensitivity_knn, ", specificity: ", specificity_knn, ", Accuracy:", accuracy_knn, "\n")
    })
    
  })
  
}

# Run the application
shinyApp(ui, server)

