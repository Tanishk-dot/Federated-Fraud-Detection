import { Route, HashRouter as Router, Routes } from "react-router-dom"
import Layout from "./components/Layout"
import Architecture from "./pages/Architecture"
import Comparison from "./pages/Comparison"
import Overview from "./pages/Overview"
import Simulation from "./pages/Simulation"
import Train from "./pages/Train"
import TestData from "./pages/TestData"

export default function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="test-data" element={<TestData />} />
          <Route path="train" element={<Train />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="comparison" element={<Comparison />} />
          <Route path="architecture" element={<Architecture />} />
        </Route>
      </Routes>
    </Router>
  )
}
