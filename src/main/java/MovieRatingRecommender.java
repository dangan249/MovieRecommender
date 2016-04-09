import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class MovieRatingRecommender {
  private String trainDataFile;
  private int numLatentFeature;
  private int numInstances;
  private int numUsers;
  private int numItems;

  private int[][] data;
  private int NUM_DATA_COLUMN = 3;
  private final int FOLD = 5;

  public MovieRatingRecommender(String trainDataFile, int numLatentFeature, int numUsers, int numItems, int numInstances) {
    this.trainDataFile = trainDataFile;
    this.numLatentFeature = numLatentFeature;
    this.numInstances = numInstances;
    this.numItems = numItems;
    this.numUsers = numUsers;

    this.data = new int[numInstances][NUM_DATA_COLUMN];
  }

  public void train() throws IOException {
    loadTrainingData();

    // we gonna slice the data in FOLD chunks and use one chunk at a time as test data
    for (int currentFold = 0; currentFold < FOLD; currentFold++) {
      System.out.println("**************");
      System.out.println("fold " + currentFold);

      int[][] ratings = getTrainingDataForFold(data, currentFold);
      int[][] testData = getTestingDataForFold(data, currentFold);

      CollaborativeFiltering collaborativeFiltering = new CollaborativeFiltering(ratings, numUsers, numItems, numLatentFeature);
      collaborativeFiltering.trainMatrixFactorization();
      collaborativeFiltering.predict(testData);
    }
  }

  private int getTestDataStartIndexForFold(int fold) {
    return fold * getNumInstancesPerFold();
  }

  private int[][] getTestingDataForFold(int[][] data, int currentFold) {
    int testDataStartIndex = getTestDataStartIndexForFold(currentFold);
    int testDataEndIndex = getTestDataEndIndexForFold(currentFold);
    int numTestRows = getNumInstancesForFold(currentFold);
    int[][] testData = new int[numTestRows][NUM_DATA_COLUMN];

    int rowIndex = 0;
    for (int i = testDataStartIndex; i <= testDataEndIndex; i++) {
      int[] row = data[i];
      for (int j = 0; j < NUM_DATA_COLUMN; j++) {
        testData[rowIndex][j] = row[j];
      }
      rowIndex++;
    }
    return testData;
  }

  private int[][] getTrainingDataForFold(int[][] data, int currentFold) {
    int testDataStartIndex = getTestDataStartIndexForFold(currentFold);
    int testDataEndIndex = getTestDataEndIndexForFold(currentFold);
    int numTestRows = getNumInstancesForFold(currentFold);
    int numRows = data.length - numTestRows;
    int leftTrainingDataStartIndex = 0;
    int leftTrainingDataEndIndex = testDataStartIndex - 1;
    int rightTrainingDataStartIndex = testDataEndIndex + 1;
    int rightTrainingDataEndIndex = data.length - 1;
    int[][] trainingData = new int[numRows][NUM_DATA_COLUMN];

    int rowIndex = 0;
    for (int i = leftTrainingDataStartIndex; i <= leftTrainingDataEndIndex; i++) {
      int[] row = data[i];
      for (int j = 0; j < NUM_DATA_COLUMN; j++) {
        trainingData[rowIndex][j] = row[j];
      }
      rowIndex++;
    }

    for (int i = rightTrainingDataStartIndex; i <= rightTrainingDataEndIndex; i++) {
      int[] row = data[i];
      for (int j = 0; j < NUM_DATA_COLUMN; j++) {
        trainingData[rowIndex][j] = row[j];
      }
      rowIndex++;
    }

    return trainingData;
  }

  private int getTestDataEndIndexForFold(int fold) {
    // check if we are at the last fold that we should get all instances up to the last index
    return fold == FOLD - 1 ? numInstances - 1 : getTestDataStartIndexForFold(fold) + getNumInstancesPerFold() - 1;
  }

  private int getNumInstancesPerFold() {
    return numInstances / FOLD;
  }

  private int getNumInstancesForFold(int fold) {
    // the last fold rarely have the same number of rows as other folds (unless the total number of instances are divided by the fold number)
    return fold == FOLD - 1 ? numInstances - (FOLD - 1) * getNumInstancesPerFold() : getNumInstancesPerFold();
  }

  private void loadTrainingData() throws IOException {
		String line;
		BufferedReader br = new BufferedReader(new FileReader(trainDataFile));
    int count = 0;
		while ((line = br.readLine()) != null) {
			String[] dataItem = line.split("\\s+");
			int user = Integer.parseInt(dataItem[0]);
			int item = Integer.parseInt(dataItem[1]);
			int rating = Integer.parseInt(dataItem[2]);

      data[count][0] = user;
      data[count][1] = item;
      data[count][2] = rating;
      count++;
		}
	}

	public static void main(String[] args) {
		int numLatenFeature = 50;
		int numUsers = 943; //943;
		int numItems = 1682;//1682;
    int numInstances = 100000;
		MovieRatingRecommender recommender = new MovieRatingRecommender("u.data", numLatenFeature, numUsers, numItems, numInstances);
		try {
			recommender.train();
		} catch (IOException e) {
			System.out.println("invalid file");
		}
	}
}
