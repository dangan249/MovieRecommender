public class CollaborativeFiltering {
  private int[][] ratings;
	private double[][] usersLatentFeatureVector;
	private double[][] itemsLatentFeatureVector;
  private int numUsers;
  private int numItems;
  private int numLatentFeature;

  private double LEARNING_RATE = 0.01;
  private double MAX_NUM_ITERATION = 5000;
  private double ACCEPTED_SQUARE_ERROR = 0.1;
  private double LAMBDA = 0.01;

  public CollaborativeFiltering(int[][] trainingData, int numUsers, int numItems, int numLatentFeature) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.numLatentFeature = numLatentFeature;

    loadRatingMatrix(trainingData);
    initializeLatentFeatureVectors();
  }

  public void trainMatrixFactorization() {
    int numIteration = 0;
    while(true) {
      for(int i = 0; i < numUsers; i++) {
        for(int j = 0; j < numItems; j++) {
          if(ratings[i][j] > 0) {
            updateUserLatentFeature(i, j);
          }
        }
      }

      // Small optimization to avoid calculating mean square error every iteration
      if(numIteration % 100 == 0) {
        //System.out.println("iteration: " + numIteration);
        double error = calculateMeanSquareError();
        //System.out.println("MSE: " + error);
        if(error < ACCEPTED_SQUARE_ERROR) break;
      }

      if(numIteration == MAX_NUM_ITERATION) {
        //System.out.println("num iteration exceeded");
        break;
      }
      numIteration++;
    }
  }


  public void predict(int[][] testData) {
    for(int i = 0;  i < testData.length; i++) {
      int[] userItemPair = testData[i];
      int userIndex = userItemPair[0] - 1;
      int itemIndex = userItemPair[1] - 1;
      int rating = userItemPair[2];
      double predictedRating = calculatePredictedRating(usersLatentFeatureVector[userIndex], itemsLatentFeatureVector[itemIndex]);
      System.out.println("Rating: " + rating + " -- Predicted: " + predictedRating);
    }
  }

  private void updateUserLatentFeature(int userIndex, int itemIndex) {
    int currentRating = ratings[userIndex][itemIndex];
    double[] currentUserLatentFeatureVector = usersLatentFeatureVector[userIndex];
    double[] currentItemLatentFeatureVector = itemsLatentFeatureVector[itemIndex];
    double newRating = calculatePredictedRating(currentUserLatentFeatureVector, currentItemLatentFeatureVector);
    double[] newUserLatentFeatureVector = new double[this.numLatentFeature];
    double[] newItemLatentFeatureVector = new double[this.numLatentFeature];

    for(int i = 0; i < this.numLatentFeature; i++) {
      newUserLatentFeatureVector[i] = currentUserLatentFeatureVector[i] + LEARNING_RATE * ((currentRating - newRating) * currentItemLatentFeatureVector[i] - LAMBDA * currentUserLatentFeatureVector[i]);
      newItemLatentFeatureVector[i] = currentItemLatentFeatureVector[i] + LEARNING_RATE * ((currentRating - newRating) * currentUserLatentFeatureVector[i] - LAMBDA * currentItemLatentFeatureVector[i]);
    }

    for(int i = 0; i < this.numLatentFeature; i++) {
      currentUserLatentFeatureVector[i] = newUserLatentFeatureVector[i];
      currentItemLatentFeatureVector[i] = newItemLatentFeatureVector[i];
    }
  }

  private double calculateMeanSquareError() {
    double error = 0;
    int numRatings = 0;
    for(int i = 0; i < numUsers; i++) {
      for (int j = 0; j < numItems; j++) {
        if(ratings[i][j] > 0) {
          double rating = ratings[i][j];
          double predictedRating = calculatePredictedRating(usersLatentFeatureVector[i], itemsLatentFeatureVector[j]);
          error += Math.pow(rating - predictedRating, 2) + LAMBDA * (calculateVectorLengthSqr(usersLatentFeatureVector[i]) + calculateVectorLengthSqr(itemsLatentFeatureVector[j]));
          numRatings++;
        }
      }
    }
    numRatings *=2;
    return error/numRatings;
  }

  private double calculateVectorLengthSqr(double[] vector) {
    double len = 0;
    for(int i = 0; i < vector.length; i++) {
      len += Math.pow(vector[i], 2);
    }
    return len;
  }

  // calculate newRating by multiple two current latent feature vectors
  private double calculatePredictedRating(double[] userLatentFeatureVector, double[] itemLatentFeatureVector) {
    double rating = 0;
    for(int i = 0; i < this.numLatentFeature; i++) {
      rating += userLatentFeatureVector[i] * itemLatentFeatureVector[i];
    }

    return rating;
  }

  private void initializeLatentFeatureVectors() {
    this.usersLatentFeatureVector = new double[numUsers][this.numLatentFeature];
    this.itemsLatentFeatureVector = new double[numItems][this.numLatentFeature];

    for(int i = 0; i < numUsers; i++) {
      for(int j = 0; j < this.numLatentFeature; j++) {
        this.usersLatentFeatureVector[i][j] = Math.random();
      }
    }


    for(int i = 0; i < numItems; i++) {
      for(int j = 0; j < this.numLatentFeature; j++) {
        this.itemsLatentFeatureVector[i][j] = Math.random();
      }
    }
  }

  private void loadRatingMatrix(int[][] trainingData) {
    this.ratings = new int[numUsers][numItems];
    for(int i = 0; i < trainingData.length; i++) {
      int[] row = trainingData[i];
      int user = row[0];
      int item = row[1];
      int rating = row[2];

      // ASSUMPTION: Users and items are numbered consecutively from 1.
      this.ratings[user-1][item-1] = rating;
    }
  }
}
