module Main where
import qualified Data.ByteString.Lazy as BS hiding (putStrLn)
import qualified Data.ByteString.Lazy as BS (putStrLn)
import Codec.Compression.GZip (decompress)
import Numeric.LinearAlgebra hiding (split)
import Prelude hiding (readFile)
import System.Random ( StdGen, mkStdGen )
import System.Random.Shuffle
import Control.Monad
import Data.List.Split
import Data.List

import Control.Monad.State
import Control.Monad (join)
import Data.Bifunctor (bimap)
import Data.ByteString.Conversion.From 
import Data.ByteString.Char8 (readInt)
import Data.Int

data Network = Network { layers  :: [Int]
                       , biases  :: [Vector R]
                       , weights :: [Matrix R]
                       , rate    :: Double
                       } deriving (Show)


main :: IO ()
main = do
  let nInputs  = 784
  let nHidden  = 30
  let nOutputs = 10
  let seed = 1
  let layers = [nInputs, nHidden, nOutputs]
  let biases = fmap (\n -> randomVector seed Gaussian n) $ drop 1 layers
  let weights = fmap (\(x, y) -> randomMatrix x y seed + 10)  $ zip (drop 1 layers) layers
  let rate = 1.5
  let net = Network layers biases weights rate

  trainingData <- readImages "images.gz" "labels.gz"
  let vec = snd $ trainingData !! 0
  --print . show . feedForward net $ vec
  print . show . fst $ trainingData !! 1
  --let input = (take 20 trainingData)
  --t <- forM input $ \(digit, input) ->
      --return (digit, train net 20 g input)


------------------- Reading MNIST -------------------------

readImages :: String -> String -> IO ([(Vector R, Vector R)])
readImages iPath lPath = do
  iBytes <- decompress <$> BS.readFile iPath
  lBytes <- decompress <$> BS.readFile lPath
  let is = readImage iBytes <$> [1..50000]
  let ls = readLabel lBytes <$> [1..50000]
  return $ zip ls is

readImage :: BS.ByteString -> Int64 -> Vector R
readImage bytes n = vector $ fmap (fromIntegral . BS.index bytes . (n*28^2 + 16 +)) [0..783]

readLabel :: BS.ByteString -> Int64 -> Vector R
readLabel bytes n = vector [if x == i then 1 else 0 | x <- [0..9]] where
                               i = fromIntegral $ BS.index bytes (n + 8)

---------------------- Training ------------------------

train :: Network-> Int -> StdGen-> [(Vector R, Vector R)] -> Network 
train net batchSize gen input = net' where
  net' = foldl' (foldl' processTrainingData) net batches
  batches = chunksOf batchSize $ shuffledInput
  shuffledInput = shuffle' input (length input) gen

processTrainingData :: Network -> (Vector R, Vector R) -> Network
processTrainingData net input = Network (layers net) bs ws (rate net) where
  ws = zipWith add (weights net) $ fmap (scale (-eta / l)) dw
  bs = zipWith add (biases net) $ fmap (scale (-eta / l)) db
  (dw, db) = unzip $ backprop net input
  l = fromIntegral $ length input
  eta = rate net

  
-- Returns nablaWeights and nableBiases in a tuple
backprop :: Network -> (Vector R, Vector R) -> [(Matrix R, Vector R)]
backprop net (y, x) = undefined where

  
  -- Propagate backwards calculating
  -- feltas thourhgout network, with input layer as head
  ds = foldl' calcDelta [outputDelta] list where
    outputDelta = cost' (head as) y * sigmoid' (head zs)
    list = init $ zip3 ws zs $ drop 1 as
    ws = reverse $ weights net
    calcDelta :: [Vector R] -> (Matrix R, Vector R, Vector R) -> [Vector R]
    calcDelta (d:ds) (w, z, a) = d':d:ds where
      d' = (tr' w #> d * sigmoid' z)

  -- Feed forward through the network
  -- calculating all zs and as, with the output layer as head
  (zs, as) = feedForward net x

  
  {-
  [(dw, db)] = foldl' calcDeltas [(calcDw d as, d)] $ init $ zip3 ws zs $ drop 1 as
ws = reverse $ weights net
  
calcDw :: Vector R -> [Vector R] -> R
  calcDw d (a:a':_) = d `dot` a'
  calcD (a:_) y = sigmoid' a * cost' a y
  
  calcDeltas :: [Vector R] -> (Matrix R, Vector R, Vector R) -> [Vector R]
  calcDeltas (d:ds) (w, z, a) = d':d:ds where
    d' = (w #> d * sigmoid' z)
  
  -- Calculate the deltas, with the input layer at index 0
  ds = foldl' calcDelta [(calcDw d as, delta as y)] $ init $ zip3 ws zs $ drop 1 as where
    calcDelta :: [Vector R] -> (Matrix R, Vector R, Vector R) -> [Vector R]
    calcDelta (d:ds) (w, z, a) = (w #> d * sigmoid' z):d:ds
    delta as y = sigmoid' a * cost' a y
    ws = reverse $ weights net
-}
  

-- Returns a tuple of zs and as throughout the network
-- with the last activation first (like a stack)
feedForward :: Network -> Vector R -> ([Vector R], [Vector R])
feedForward net a = foldl' ffd ([a], []) $ zip (weights net) (biases net)  where
  ffd :: ([Vector R], [Vector R]) -> (Matrix R, Vector R) -> ([Vector R], [Vector R])
  ffd (a:as, zs) (w, b) = (a':a:as, z:zs) where
    a' = sigmoid z
    z  = w #> a + b


{-
-- Perform stochastic gradient descent on the batch
-- and return a new network with updated weights and biases
processBatch :: Network -> [(Vector R, Vector R)] -> Network
processBatch net batch = foldl' processTrainingData net batch where
  processTrainingData :: Network -> (Vector R, Vector R) -> Network
  processTrainingData = \net input ->
    Network (layers net) bs ws (rate net) where
      ws = zipWith add (weights net) $ fmap (scale (-eta / l)) dw
      bs = zipWith add (biases net) $ fmap (scale (-eta / l)) db
      (dw, db) = unzip $ backprop net input
      l = fromIntegral $ length batch
      eta = rate net
-}

-- Returns a tuple of zs and as throughout the network
-- with the last activation first (like a stack)
{-
oldfeedForward :: Network -> Vector R -> ([Vector R], [Vector R])
oldfeedForward net a = ffd (weights net) (biases net) ([a], []) where
  ffd :: [Matrix R] -> [Vector R] -> ([Vector R], [Vector R]) -> ([Vector R], [Vector R])
  ffd [] [] a = a
  ffd (w:ws) (b:bs) (a:as, zs) = ffd ws bs (a':a:as, z:zs) where
    a' = sigmoid z
    z  = w #> a + b
-}
  -- Do the backpropagation
--forM ds $ \(lbl, img) ->
--                                    let a = feedForward n img
                                     
-- Cost function
cost w b n xs a = 1/(2*n) * (sum [(norm_2 (y_ x - a))^2 | x <- xs]) 

-- Derivative of cost
cost' :: Vector R -> Vector R -> Vector R
cost' a y = a - y

sigmoid :: Floating a => a -> a
sigmoid z = 1/(1 + exp(z))

sigmoid' :: Floating a => a -> a
sigmoid' z = sigmoid(z)*(1 - sigmoid(z))


--------------------  Utility   ---------------------------

randomMatrix :: Int -> Int -> Int -> Matrix R
randomMatrix x y seed = (x><y) . toList . randomVector seed Gaussian $ x * y


y_ :: Vector R -> Vector R
y_ = undefined

-- Test stuff
{-
getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]

render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)
pasta = do
  s <- decompress <$> BS.readFile "images.gz"
  l <- decompress <$> BS.readFile "labels.gz"
  let n = 1

  -- Labels 8 bytes
  -- Images has offset 16 8 * 28  28?

  --n <- (`mod` 60000) <$> randomIO
  let t =  (getImage s n :: [Double])
  putStrLn $ show t
  {-
  putStr . unlines $
    [(render . BS.index s . (n*28^2 + 16 + r*28 +)) <$> [0..27] | r <- [0..27]]
  print $ BS.index l (n + 8)
-}
-}
